"""Build one registered neural-run configuration from explicit, independently pinned inputs."""
from .contract import BRANCHES, Dataset, HASH, SEEDS, digest, output_dir, preprocessing, put, read, require


def config(cfg, phase, lr_path=None):
    branch,seed=cfg["branch"],cfg["seed"]
    require(cfg["schema"]=="el-vision-run-1" and cfg["phase"]==phase and branch in BRANCHES and
            type(seed) is int and seed in SEEDS and type(cfg["epochs"]) is int and 1<=cfg["epochs"]<=20
            and cfg["patience"]==3 and cfg["microbatch"]==1 and cfg["accumulate"]==4
            and cfg["weight_decay"]==.01 and cfg["precision"]=="float32",
            "training budget, precision, seed or branch differs from the preregistered design")
    expected=(cfg["backbone_lr"]==0 and cfg["head_lr"] in (1e-4,3e-4)) if branch=="L2-frozen" else (
        cfg["backbone_lr"] in (1e-5,3e-5) and cfg["head_lr"]==3e-4)
    require(expected, "unregistered learning rate")
    if phase=="pilot":
        require(seed==SEEDS[0] and type(cfg["max_updates"]) is int and 1<=cfg["max_updates"]<=32
                and cfg.get("development_dataset_sha256") is None,"pilot uses train only and at most 32 updates")
    else:
        require(HASH.fullmatch(cfg["development_dataset_sha256"]) and cfg.get("max_updates") is None,
                "formal fit requires development and epoch stopping")
    if phase=="train" and seed!=SEEDS[0]:
        require(lr_path and digest(lr_path)==cfg.get("lr_selection_sha256"), "repeat needs frozen seed-A LR selection")
        choice=read(lr_path)
        require(choice["schema"]=="el-vision-lr-selection-1" and choice["seed"]==SEEDS[0]
                and choice["training_dataset_sha256"]==cfg["training_dataset_sha256"]
                and choice["development_dataset_sha256"]==cfg["development_dataset_sha256"]
                and choice["preprocessing_sha256"]==cfg["preprocessing_sha256"]
                and choice["base_pin_sha256"]==cfg["base_pin_sha256"]
                and all(choice["chosen"][branch][k]==cfg[k] for k in ("head_lr","backbone_lr")),
                "repeat does not use its paired selected configuration")


def configure(a):
    out=output_dir(a.output)
    require(digest(a.catalog)==a.catalog_sha256 and digest(a.preprocessing)==a.preprocessing_sha256
            and digest(a.base_pin)==a.base_pin_sha256,"catalog, preprocessing or base pin differs")
    cat=read(a.catalog)
    train=Dataset(a.train,"train",a.train_sha256)
    pre=preprocessing(a.preprocessing,train_sha=train.sha,feature_sha=train.schema)
    require(set(pre["family_ids"])==set(train.families),"preprocessing was fit on a different family roster")
    doc={"schema":"el-vision-run-1","phase":a.phase,"branch":a.branch,"seed":a.seed,"backbone_lr":a.backbone_lr,
         "head_lr":a.head_lr,"epochs":a.epochs,"patience":3,"microbatch":1,"accumulate":4,"weight_decay":.01,
         "precision":"float32","training_dataset_sha256":train.sha,"development_dataset_sha256":None,
         "preprocessing_sha256":a.preprocessing_sha256,"base_pin_sha256":a.base_pin_sha256,
         "catalog_sha256":a.catalog_sha256,"training_family_ids":sorted(train.families)}
    if a.phase=="pilot":
        require(not a.development and not a.development_sha256 and set(cat["pilot_families"])==set(train.families) and
                len(cat["pilot_families"])==4 and pre["scope"]=="engineering_pilot","pilot requires exactly four registered train families")
        doc.update(pilot_families=sorted(train.families),max_updates=a.max_updates or 32)
    else:
        require(a.development and a.development_sha256 and not a.max_updates and pre["scope"]=="specified_materials",
                "formal runs require original train/dev datasets and a train-only normalization")
        dev=Dataset(a.development,"development",a.development_sha256)
        new_train={r["scene_family_id"] for r in cat["families"] if r["split"]=="train"}
        new_dev={r["scene_family_id"] for r in cat["families"] if r["split"]=="development"}
        require(len(train.families)==28 and len(dev.families)==8 and new_train<=train.families.keys() and
                new_dev<=dev.families.keys() and not train.families.keys() & dev.families.keys(),
                "formal cohort must include the registered new and historical train/development families")
        doc.update(development_dataset_sha256=dev.sha,development_family_ids=sorted(dev.families))
    if a.lr_selection:
        require(a.lr_selection_sha256 and digest(a.lr_selection)==a.lr_selection_sha256,"learning rate selection differs")
        doc["lr_selection_sha256"]=a.lr_selection_sha256
    config(doc,a.phase,a.lr_selection)
    put(out/"config.json",doc)
    return doc
