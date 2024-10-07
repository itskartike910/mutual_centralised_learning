import argparse
import os
import os.path as osp
from configs.miniimagenet_default import cfg

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", '--device', type=str, dest='device', default='0')
    args = parser.parse_args()

    config_file_path = './configs/miniImagenet/pretrainer/FRN_pre.yaml'

    dataset = cfg_to_dataset(config_file_path)
    if not hasattr(args, 'prefix') or not args.prefix:
        args.prefix = dataset + "_" + osp.basename(config_file_path).replace(".yaml", "")

    cfg.merge_from_file(config_file_path)
    if hasattr(args, 'rest') and args.rest:
        cfg.merge_from_list(args.rest)
    cfg.data.image_dir = osp.join(cfg.data.root, dataset)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    checkpoint_dir = osp.join(args.checkpoint_base, args.prefix)
    snapshot_dir = osp.join(args.snapshot_base, args.prefix)
    for d in [checkpoint_dir, snapshot_dir]:
        if not osp.exists(d):
            os.mkdir(d)
    print("[*] Source Image Path: {}".format(cfg.data.image_dir))
    print("[*] Target Checkpoint Path: {}".format(checkpoint_dir))

    trainer = t(cfg, checkpoint_dir)
    trainer.run()

    shutil.copyfile(config_file_path, osp.join(snapshot_dir, osp.basename(config_file_path)))
    shutil.copyfile(trainer.snapshot_for_meta, osp.join(snapshot_dir, osp.basename(trainer.snapshot_for_meta)))
    shutil.copytree(trainer.writer_dir, osp.join(snapshot_dir, osp.basename(trainer.writer_dir)))

if __name__ == "__main__":
    main()