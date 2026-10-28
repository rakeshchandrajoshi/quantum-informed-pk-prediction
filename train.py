"""Train independent 10-seed models using held-out official benchmark partitions.

Only the *internal training-fold* is used to estimate normalization statistics or
select epochs. The official test set is evaluated after validation selection.
"""
from __future__ import annotations

import argparse
import hashlib
import platform
import copy
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn,Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from .data import (FeatureImputerScaler, NumericDataset, GraphExample, collate_graphs,
                   load_graph_features, load_numeric_features)
from .metrics import summarize
from .models import FACTORS, ConventionalMLP, GraphMPNN, MFMN, count_parameters


def default_config(model: str, endpoint: str) -> dict[str, Any]:
    if model == "graphmpnn":
        if endpoint not in {"pka_acidic", "pka_basic"}:
            raise ValueError("GraphMPNN is configured for pka_acidic and pka_basic")
        return dict(lr=5e-4, weight_decay=1e-5, batch_size=64, max_epochs=300,
                    patience=40, grad_clip=5.0, val_fraction=.15, scaling="none")
    if model not in {"mfmn", "mlp"} or endpoint not in {"CL","VDss","Fu"}:
        raise ValueError("MFMN/MLP require a CL, VDss, or Fu endpoint")
    return dict(lr=3e-3,weight_decay=3e-2 if endpoint=="Fu" else 1e-2,
                batch_size=128,max_epochs=400,patience=50,grad_clip=1.0,
                val_fraction=.15,scaling="none")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def split_train_for_validation(splits: np.ndarray, seed: int,
                               fraction: float) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    if not (0 < fraction < 0.5):
        raise ValueError("Internal validation fraction must lie in (0,0.5)")
    all_train = np.flatnonzero(splits=="train")
    test = np.flatnonzero(splits=="test")
    if len(all_train) < 4 or len(test) < 2:
        raise ValueError("At least four training and two test compounds are required")
    shuffled = np.random.default_rng(seed).permutation(all_train)
    count = max(1,int(np.ceil(len(shuffled)*fraction)))
    return shuffled[count:], shuffled[:count], test


def _device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(spec)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but no CUDA device is available")
    return device


def _loader(ds, batch_size:int, shuffle:bool, seed:int, *, collate_fn=None):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(ds,batch_size=batch_size,shuffle=shuffle,collate_fn=collate_fn,
                      generator=generator, num_workers=0)


def _numeric_dataloader(features:dict[str,Tensor], aux:dict[str,Tensor],
                        targets:np.ndarray, idx:np.ndarray, norm:tuple[float,float],
                        batch:int, shuffle:bool, seed:int, endpoint:str):
    tensors = [features["context"],*[features[name] for name in FACTORS]]
    ynorm = torch.tensor((targets[idx]-norm[0])/norm[1],dtype=torch.float32)
    # Aux labels are masked, standardized separately, and never NaN in loss.
    aux_vals=[]
    aux_masks=[]
    for name in FACTORS:
        if endpoint == "Fu":
            v = torch.zeros(len(idx))
            m = torch.zeros(len(idx),dtype=torch.bool)
        else:
            values=aux[name][idx]
            m=torch.tensor(np.isfinite(values))
            v=torch.nan_to_num(torch.tensor(values,dtype=torch.float32),nan=0.0)
        aux_vals.append(v)
        aux_masks.append(m)
    weight_values=(np.where(targets[idx] < np.log10(.006),1.25,1.0)
                   if endpoint=="Fu" else np.ones(len(idx)))
    weights=torch.tensor(weight_values,dtype=torch.float32)
    dataset=TensorDataset(*tensors,ynorm,*aux_vals,*aux_masks,weights)
    return _loader(dataset,batch,shuffle,seed)


def _numeric_batches(dataset:NumericDataset, fit_idx:np.ndarray, val_idx:np.ndarray,
                     test_idx:np.ndarray, config:dict, endpoint:str, seed:int):
    raw={"context":dataset.context,**dataset.factors}
    scaling = FeatureImputerScaler(config["scaling"])
    scaling.fit(raw,fit_idx)
    y=dataset.targets
    mean=float(np.mean(y[fit_idx])); sd=float(np.std(y[fit_idx])); sd=max(sd,1e-9)
    target_norm=(mean,sd)
    aux_norm = {}
    normalized_aux={}
    for name in FACTORS:
        if endpoint=="Fu":
            continue
        raw_aux=dataset.aux[name]
        valid=raw_aux[fit_idx][np.isfinite(raw_aux[fit_idx])]
        if not len(valid):
            # Entirely missing auxiliary target; mask loss for this factor.
            amean,astd=0.0,1.0
        else:
            amean,astd=float(np.mean(valid)),max(float(np.std(valid)),1e-9)
        aux_norm[name]={"mean":amean,"std":astd}
        normalized_aux[name]=(raw_aux-amean)/astd
    loaders={}
    for label, idx in (("train",fit_idx),("validation",val_idx),("test",test_idx)):
        x=scaling.transform(raw,idx)
        loaders[label]=_numeric_dataloader(x,normalized_aux,y,idx,target_norm,
                                          config["batch_size"],label=="train",seed,
                                          endpoint)
    return loaders,target_norm,scaling.fitted,aux_norm


def _numeric_forward(model:nn.Module, batch:list[Tensor], kind:str,endpoint:str,
                    device:torch.device,aux_lambda:float)->tuple[Tensor,Tensor]:
    data=[x.to(device) for x in batch]
    ctx,*remaining=data
    factortensors=remaining[:7]
    labels=remaining[7]
    aux_targets=remaining[8:15]
    aux_masks=remaining[15:22]
    weights=remaining[22]
    if kind=="mfmn":
        predictions,aux_outputs=model(ctx,{name:arr for name,arr in zip(FACTORS,factortensors)})
    else:
        predictions=model(torch.cat([ctx,*factortensors],dim=-1))
        aux_outputs={}
    primary=F.smooth_l1_loss(predictions,labels,beta=.5,reduction="none")
    # Fu tail weighting uses original log10 target before standardization.
    loss=(primary*weights).sum()/weights.sum()
    if kind=="mfmn" and endpoint in {"CL","VDss"}:
        for name,label,mask in zip(FACTORS,aux_targets,aux_masks):
            if bool(mask.any().item()):
                loss=loss+aux_lambda*(aux_outputs[name][mask]-label[mask]).square().mean()
    return loss,predictions


def _graph_forward(model:nn.Module,batch,device:torch.device)->tuple[Tensor,Tensor]:
    moved=[item.to(device) for item in batch]
    *inputs,y=moved
    pred=model(*inputs)
    return F.smooth_l1_loss(pred,y,beta=.5),pred


def _train_one(model,loaders:dict, *, model_kind:str,endpoint:str, config:dict,
               device:torch.device,meanstd:tuple[float,float],train_seed:int):
    opt=torch.optim.AdamW(model.parameters(),lr=config["lr"],weight_decay=config["weight_decay"])
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=config["max_epochs"])
    model.to(device)
    best=float("inf");best_epoch=0;state=None; epochs_done=0
    aux_lambda={"CL":.8,"VDss":2.0,"Fu":0.0}.get(endpoint,0)
    for epoch in range(1,config["max_epochs"]+1):
        model.train()
        for batch in loaders["train"]:
            opt.zero_grad(set_to_none=True)
            if model_kind=="graphmpnn":
                loss,_=_graph_forward(model,batch,device)
            else:
                loss,_=_numeric_forward(model,batch,model_kind,endpoint,device,aux_lambda)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),config["grad_clip"])
            opt.step()
        model.eval(); total=0.0; n=0
        with torch.no_grad():
            for batch in loaders["validation"]:
                if model_kind=="graphmpnn":
                    loss,_=_graph_forward(model,batch,device)
                else:
                    loss,_=_numeric_forward(model,batch,model_kind,endpoint,device,aux_lambda)
                batchlen=len(batch[0]);total+=float(loss.item())*batchlen;n+=batchlen
        val_loss=total/max(n,1)
        if not np.isfinite(val_loss):
            raise RuntimeError("Non-finite validation loss")
        if val_loss < best-1e-10:
            best=val_loss;best_epoch=epoch;state=copy.deepcopy(model.state_dict())
        schedule.step()
        epochs_done=epoch
        if epoch-best_epoch >= config["patience"]:
            break
    assert state is not None
    model.load_state_dict(state)
    model.eval()
    prediction_parts=[]
    with torch.no_grad():
        for batch in loaders["test"]:
            if model_kind=="graphmpnn":
                _,pred=_graph_forward(model,batch,device)
            else:
                _,pred=_numeric_forward(model,batch,model_kind,endpoint,device,aux_lambda)
            prediction_parts.append(pred.cpu().numpy())
    normalized_pred=np.concatenate(prediction_parts)
    return normalized_pred*meanstd[1]+meanstd[0],{"best_epoch":best_epoch,"epochs_run":epochs_done,
                                                "best_validation_loss":best},model


def run_training(*, input_path:str,output_dir:str,model_kind:str,endpoint:str,
                 seeds:list[int],overrides:dict[str,Any]|None=None,device_spec:str="auto",
                 strict_dimensions:bool=True) -> dict[str,Any]:
    """Run selected architecture; write checkpoints + test predictions + evaluation."""
    config=default_config(model_kind,endpoint)
    if overrides:
        unknown=set(overrides)-set(config)
        if unknown:raise ValueError(f"Unknown training overrides: {sorted(unknown)}")
        config.update(overrides)
    if not seeds or len(set(seeds))!=len(seeds):
        raise ValueError("Seeds must be a nonempty list of unique integers")
    if int(config["max_epochs"])<1 or int(config["batch_size"])<1 or int(config["patience"])<1:
        raise ValueError("Invalid epoch/batch/patience")
    destination=Path(output_dir)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError("Output directory is not empty; refusing to overwrite experiments")
    destination.mkdir(parents=True,exist_ok=True)
    device=_device(device_spec)
    if model_kind=="graphmpnn":
        data=load_graph_features(input_path)
        splits=np.asarray([r.split for r in data]);ids=[r.compound_id for r in data]
        targets=np.asarray([r.target for r in data])
    else:
        data=load_numeric_features(input_path,endpoint=endpoint,enforce_manuscript_dims=strict_dimensions)
        splits=data.splits; ids=data.compounds;targets=data.targets
    per_seed=[]; outputs=[];architecture_counts=[]
    for seed in seeds:
        set_seed(seed)
        fit_idx,val_idx,test_idx=split_train_for_validation(splits,seed,config["val_fraction"])
        mean=float(np.mean(targets[fit_idx]));std=max(float(np.std(targets[fit_idx])),1e-9)
        if model_kind=="graphmpnn":
            model=GraphMPNN(site_readout=(endpoint=="pka_acidic"))
            subsets={k:[data[i] for i in idx] for k,idx in (("train",fit_idx),
                                                ("validation",val_idx),("test",test_idx))}
            loaders={k:_loader(v,config["batch_size"],k=="train",seed,
                               collate_fn=lambda examples,m=(mean,std):collate_graphs(examples,m))
                     for k,v in subsets.items()}
            featstate={"prepared_graph_inputs": "36 atomic, 8 bond", "external_chemistry_features": True}
            auxstate={}
        else:
            dims={name:len(data.factor_columns[name]) for name in FACTORS}
            model=(MFMN(data.context.shape[1],dims,endpoint=endpoint) if model_kind=="mfmn"
                   else ConventionalMLP(data.context.shape[1]+sum(dims.values()),endpoint=endpoint))
            loaders,_,scaler_state,auxstate=_numeric_batches(data,fit_idx,val_idx,test_idx,config,endpoint,seed)
            featstate={"context_columns":data.context_columns,"factor_columns":data.factor_columns,
                       "feature_normalization":scaler_state}
        parameter_count=count_parameters(model)
        architecture_counts.append(parameter_count)
        y_pred,history,model=_train_one(model,loaders,model_kind=model_kind,endpoint=endpoint,
                                       config=config,device=device,meanstd=(mean,std),train_seed=seed)
        y_test=targets[test_idx]
        scores=summarize(y_test,y_pred,endpoint)
        for i,pred in zip(test_idx,y_pred):
            outputs.append(dict(compound_id=ids[i],endpoint=endpoint,seed=seed,
                                y_true=float(targets[i]),y_pred=float(pred)))
        per_seed.append(dict(seed=seed,**scores,**history))
        torch.save({"model_state":model.cpu().state_dict(),"model_kind":model_kind,
                    "endpoint":endpoint,"config":config,"feature_state":featstate,
                    "auxiliary_target_scalers":auxstate,"target_mean":mean,
                    "target_std":std,"parameters":parameter_count},destination/f"seed_{seed}.pt")
        print(f"{model_kind}/{endpoint} seed {seed}: {history['epochs_run']} epochs; "
              f"best epoch {history['best_epoch']}; test MAE {scores['MAE']:.4f}")
    seed_frame=pd.DataFrame(outputs)
    seed_frame=seed_frame.sort_values(["compound_id","seed"])
    seed_frame.to_csv(destination/"seed_predictions.csv",index=False)
    # Verify same held-out compound and reference label across all seeds.
    g=seed_frame.groupby("compound_id",sort=True)
    if any(len(grp)!=len(seeds) or grp.y_true.nunique()!=1 for _,grp in g):
        raise AssertionError("Compound/target alignment failed during ensembling")
    ensemble=g.agg(endpoint=("endpoint","first"),y_true=("y_true","first"),
                   y_pred=("y_pred","mean")).reset_index()
    ensemble.to_csv(destination/"ensemble_predictions.csv",index=False)
    pd.DataFrame(per_seed).to_csv(destination/"seed_metrics.csv",index=False)
    ensemble_scores=summarize(ensemble.y_true,ensemble.y_pred,endpoint)
    hasher=hashlib.sha256()
    with Path(input_path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024),b""):
            hasher.update(chunk)
    summary={"implementation":"independent_reference_implementation",
             "input_sha256":hasher.hexdigest(),"python_version":platform.python_version(),
             "pytorch_version":torch.__version__,"numpy_version":np.__version__,
             "reproduces_reported_paper_results":False,"model_kind":model_kind,
             "endpoint":endpoint,"input_path":str(input_path),"seeds":seeds,
             "configuration":config,"parameter_count":architecture_counts[0],
             "fit_policy":"15% internal seed-specific validation from official training; official test only after selection",
             "n_official_train":int(np.sum(splits=="train")),
             "n_official_test":int(np.sum(splits=="test")),
             "ensemble_metrics":ensemble_scores}
    (destination/"run_metadata.json").write_text(json.dumps(summary,indent=2)+"\n")
    print("Ensemble:",json.dumps(ensemble_scores,indent=2))
    return summary


def main() -> None:
    parser=argparse.ArgumentParser(description="Train GraphMPNN/MFMN or matched conventional MLP")
    parser.add_argument("--input",required=True,help="Prepared CSV (MFMN/MLP) or JSONL (GraphMPNN)")
    parser.add_argument("--output",required=True,help="New or empty output directory")
    parser.add_argument("--model",required=True,choices=["mfmn","mlp","graphmpnn"])
    parser.add_argument("--endpoint",required=True,choices=["CL","VDss","Fu","pka_acidic","pka_basic"])
    parser.add_argument("--seeds",default="42,43,44,45,46,47,48,49,50,51")
    parser.add_argument("--config",help="Optional configuration JSON containing model, endpoint and settings")
    parser.add_argument("--device",default="auto")
    parser.add_argument("--max-epochs",type=int)
    parser.add_argument("--batch-size",type=int)
    parser.add_argument("--feature-scaling",choices=["none","zscore"])
    parser.add_argument("--allow-nonbenchmark-dimensions",action="store_true",
                        help="Only for testing alternate features; disables 199/19, 1419/46 checks")
    args=parser.parse_args()
    overrides={}
    if args.config:
        config_json=json.loads(Path(args.config).read_text())
        if config_json.get("model")!=args.model or config_json.get("endpoint")!=args.endpoint:
            parser.error("Configuration model/endpoint must match CLI arguments")
        overrides.update(config_json.get("settings",{}))
    if args.max_epochs is not None:overrides["max_epochs"]=args.max_epochs
    if args.batch_size is not None:overrides["batch_size"]=args.batch_size
    if args.feature_scaling is not None:overrides["scaling"]=args.feature_scaling
    run_training(input_path=args.input,output_dir=args.output,model_kind=args.model,
                 endpoint=args.endpoint,seeds=[int(i) for i in args.seeds.split(",")],
                 overrides=overrides,device_spec=args.device,
                 strict_dimensions=not args.allow_nonbenchmark_dimensions)

if __name__=="__main__":
    main()
