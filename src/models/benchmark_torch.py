import os
import argparse
import time
from contextlib import contextmanager
from PIL import Image

from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Subset

from oodeel.methods import MLS, Energy, Entropy, DKNN, Gram, Mahalanobis, ODIN, VIM
from oodeel.eval.metrics import bench_metrics
from oodeel.datasets import OODDataset
from oodeel.types import List


# args
parser = argparse.ArgumentParser(description="Benchmarking OOD detection methods")
parser.add_argument(
    "--cuda", type=int, default=0, help="Index of the CUDA device to use (default: 0)"
)
parser.add_argument(
    "--cpu", action="store_true", help="Use CPU instead of GPU (default: False)"
)
args = parser.parse_args()

# # device
device = torch.device("cpu" if args.cpu else f"cuda:{args.cuda}")
print("Using device:", device)

# paths
model_path = os.path.expanduser("~/") + ".oodeel/saved_models"
data_path = os.path.expanduser("~/") + ".oodeel/datasets"
os.makedirs(model_path, exist_ok=True)
os.makedirs(data_path, exist_ok=True)


# utils
class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.files = os.listdir(root)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.root, self.files[idx])).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, 0


# models
def load_model(experiment="mnist", load_mlp=False):
    # model trained on MNIST[0-4]
    if experiment == "mnist":
        if load_mlp:
            model_path_mnist_04 = os.path.join(model_path, "mnist_mlp_0-4")
        else:
            model_path_mnist_04 = os.path.join(model_path, "mnist_model_0-4")
        model = torch.load(
            os.path.join(model_path_mnist_04, "best.pt"), map_location=device
        )
    # model trained on CIFAR10
    elif experiment == "cifar10":
        model = torch.hub.load(
            repo_or_dir="chenyaofo/pytorch-cifar-models",
            model="cifar10_resnet20",
            pretrained=True,
            verbose=False,
        ).to(device)
    # model trained on ImageNet
    elif experiment == "imagenet":
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1
        model = torchvision.models.resnet50(weights=weights).to(device)
    else:
        raise ValueError("`experiment` should be 'mnist', 'cifar10' or 'imagenet'.")
    model.eval()
    return model


# datasets
def load_datasets(experiment: str = "mnist", batch_size: int = 128):
    # MNIST[0-4] vs MNIST[5-9]
    if experiment == "mnist":
        # 1- load train/test MNIST dataset
        mnist_train = OODDataset(
            dataset_id="MNIST",
            backend="torch",
            load_kwargs={"root": data_path, "train": True, "download": True},
        )
        mnist_test = OODDataset(
            dataset_id="MNIST",
            backend="torch",
            load_kwargs={"root": data_path, "train": False, "download": True},
        )

        # 2- split ID / OOD data depending on label value:
        # in-distribution: MNIST[0-4] / out-of-distribution: MNIST[5-9]
        in_labels = [0, 1, 2, 3, 4]
        oods_fit, _ = mnist_train.split_by_class(in_labels=in_labels)
        oods_in, oods_out = mnist_test.split_by_class(in_labels=in_labels)
        oods_out_dict = {"mnist5-9": oods_out}

        # 3- preprocess function
        def preprocess_fn(*inputs):
            """Simple preprocessing function to normalize images in [0, 1].

            Args:
                *inputs: inputs to preprocess.
            """
            x = inputs[0] / 255.0
            return tuple([x] + list(inputs[1:]))

    # CIFAR10 vs SVHN
    elif experiment == "cifar10":
        # 1a- load in-distribution dataset: CIFAR-10
        oods_fit = OODDataset(
            dataset_id="CIFAR10",
            backend="torch",
            load_kwargs={"root": data_path, "train": True, "download": True},
        )
        oods_in = OODDataset(
            dataset_id="CIFAR10",
            backend="torch",
            load_kwargs={"root": data_path, "train": False, "download": True},
        )
        # 1b- load out-of-distribution dataset: SVHN
        oods_out = OODDataset(
            dataset_id="SVHN",
            backend="torch",
            load_kwargs={"root": data_path, "split": "test", "download": True},
        )
        oods_out_dict = {"svhn": oods_out}

        # 2- preprocess function
        def preprocess_fn(*inputs):
            """Preprocessing function from
            https://github.com/chenyaofo/pytorch-cifar-models
            """
            x = inputs[0] / 255.0
            x = transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            )(x)
            return tuple([x] + list(inputs[1:]))

    elif experiment == "imagenet":
        max_samples = 50_000

        # 1a- load in-distribution dataset: ImageNet
        imagenet_root = "/local_data/imagenet_cache/ILSVRC/Data/CLS-LOC"

        # === train (calibration) ===
        imagenet_train = torchvision.datasets.ImageFolder(
            root=os.path.join(imagenet_root, "train"),
        )
        print("len(imagenet_train):", len(imagenet_train))
        g = torch.Generator().manual_seed(42)
        indices_imagenet_train = torch.randperm(len(imagenet_train), generator=g)[
            :max_samples
        ]
        oods_fit = OODDataset(
            dataset_id=Subset(imagenet_train, indices=indices_imagenet_train),
            backend="torch",
        )

        # === val (ID) ===
        imagenet_val = torchvision.datasets.ImageFolder(
            root=os.path.join(imagenet_root, "val"),
        )
        print("len(imagenet_val):", len(imagenet_val))
        g = torch.Generator().manual_seed(43)
        indices_imagenet_val = torch.randperm(len(imagenet_val), generator=g)[
            :max_samples
        ]
        oods_in = OODDataset(
            dataset_id=Subset(
                imagenet_val,
                indices=indices_imagenet_val,
            ),
            backend="torch",
        )

        # 1b- load out-of-distribution dataset
        # === textures ===
        textures_root = "/datasets/openood/images_classic/texture"
        textures = torchvision.datasets.ImageFolder(root=textures_root)
        print("len(textures):", len(textures))
        g = torch.Generator().manual_seed(44)
        indices_textures = torch.randperm(len(textures), generator=g)[:max_samples]
        oods_out_textures = OODDataset(
            dataset_id=Subset(
                textures,
                indices=indices_textures,
            ),
            backend="torch",
        )

        # === inaturalist ===
        inaturalist_root = "/datasets/openood/images_largescale/inaturalist/images"
        inaturalist = SimpleDataset(root=inaturalist_root)
        print("len(inaturalist):", len(inaturalist))
        g = torch.Generator().manual_seed(45)
        indices_inaturalist = torch.randperm(len(inaturalist), generator=g)[
            :max_samples
        ]
        oods_out_inaturalist = OODDataset(
            dataset_id=Subset(
                inaturalist,
                indices=indices_inaturalist,
            ),
            backend="torch",
        )

        # === openimage-o ===
        openimage_root = "/datasets/openood/images_largescale/openimage_o/images"
        openimage = SimpleDataset(root=openimage_root)
        print("len(openimage):", len(openimage))
        g = torch.Generator().manual_seed(46)
        indices_openimage = torch.randperm(len(openimage), generator=g)[:max_samples]
        oods_out_openimage = OODDataset(
            dataset_id=Subset(
                openimage,
                indices=indices_openimage,
            ),
            backend="torch",
        )

        # === ssb-hard ===
        ssb_hard_root = "/datasets/openood/images_largescale/ssb_hard"
        ssb_hard = torchvision.datasets.ImageFolder(root=ssb_hard_root)
        print("len(ssb_hard):", len(ssb_hard))
        g = torch.Generator().manual_seed(47)
        indices_ssb_hard = torch.randperm(len(ssb_hard), generator=g)[:max_samples]
        oods_out_ssb_hard = OODDataset(
            dataset_id=Subset(
                ssb_hard,
                indices=indices_ssb_hard,
            ),
            backend="torch",
        )

        # === ninco ===
        ninco_root = "/datasets/openood/images_largescale/ninco"
        ninco = torchvision.datasets.ImageFolder(root=ninco_root)
        print("len(ninco):", len(ninco))
        g = torch.Generator().manual_seed(48)
        indices_ninco = torch.randperm(len(ninco), generator=g)[:max_samples]
        oods_out_ninco = OODDataset(
            dataset_id=Subset(
                ninco,
                indices=indices_ninco,
            ),
            backend="torch",
        )

        oods_out_dict = {
            "ssb-hard": oods_out_ssb_hard,
            "ninco": oods_out_ninco,
            "textures": oods_out_textures,
            "inaturalist": oods_out_inaturalist,
            "openimage-o": oods_out_openimage,
        }

        # 2- preprocess function
        def preprocess_fn(*inputs):
            """ImageNet preprocessing function from torchvision.

            Args:
                *inputs: inputs to preprocess.
            """
            x = torchvision.models.ResNet50_Weights.IMAGENET1K_V1.transforms()(
                inputs[0]
            )
            return tuple([x] + list(inputs[1:]))

    else:
        raise ValueError("`experiment` should be 'mnist' or 'cifar10'.")

    # prepare dataloaders
    ds_fit = oods_fit.prepare(
        batch_size=batch_size, preprocess_fn=preprocess_fn, shuffle=False
    )
    ds_in = oods_in.prepare(batch_size=batch_size, preprocess_fn=preprocess_fn)
    ds_out_dict = {
        name: ood_out.prepare(batch_size=batch_size, preprocess_fn=preprocess_fn)
        for (name, ood_out) in oods_out_dict.items()
    }
    return ds_fit, ds_in, ds_out_dict


@contextmanager
def Timer():
    """Timer context manager.

    Yields:
        lambda: time.time() - t0
    """
    t0 = time.time()
    yield lambda: time.time() - t0


class BenchmarkTorch:
    REACT_DETECTORS = ["MLS", "MSP", "Energy", "Entropy", "ODIN"]
    DETECTORS_CONFIG = {
        "MLS": {
            "class": MLS,
            "kwargs": {
                "mnist": dict(),
                "cifar10": dict(),
                "imagenet": dict(),
            },
            "fit_kwargs": {
                "mnist": dict(),
                "cifar10": dict(),
                "imagenet": dict(),
            },
        },
        "MSP": {
            "class": MLS,
            "kwargs": {
                "mnist": dict(output_activation="softmax"),
                "cifar10": dict(output_activation="softmax"),
                "imagenet": dict(output_activation="softmax"),
            },
            "fit_kwargs": {
                "mnist": dict(),
                "cifar10": dict(),
                "imagenet": dict(),
            },
        },
        "Energy": {
            "class": Energy,
            "kwargs": {
                "mnist": dict(),
                "cifar10": dict(),
                "imagenet": dict(),
            },
            "fit_kwargs": {
                "mnist": dict(),
                "cifar10": dict(),
                "imagenet": dict(),
            },
        },
        "Entropy": {
            "class": Entropy,
            "kwargs": {
                "mnist": dict(),
                "cifar10": dict(),
                "imagenet": dict(),
            },
            "fit_kwargs": {
                "mnist": dict(),
                "cifar10": dict(),
                "imagenet": dict(),
            },
        },
        "ODIN": {
            "class": ODIN,
            "kwargs": {
                "mnist": dict(temperature=1000),
                "cifar10": dict(temperature=1000),
                "imagenet": dict(temperature=1000),
            },
            "fit_kwargs": {
                "mnist": dict(),
                "cifar10": dict(),
                "imagenet": dict(),
            },
        },
        "DKNN": {
            "class": DKNN,
            "kwargs": {
                "mnist": dict(nearest=50),
                "cifar10": dict(nearest=50),
                "imagenet": dict(nearest=50),
            },
            "fit_kwargs": {
                "mnist": dict(feature_layers_id=[-2]),
                "cifar10": dict(feature_layers_id=[-2]),
                "imagenet": dict(feature_layers_id=[-2]),
            },
        },
        "Mahalanobis": {
            "class": Mahalanobis,
            "kwargs": {
                "mnist": dict(),
                "cifar10": dict(),
                "imagenet": dict(),
            },
            "fit_kwargs": {
                "mnist": dict(feature_layers_id=[-2]),
                "cifar10": dict(feature_layers_id=[-2]),
                "imagenet": dict(feature_layers_id=[-2]),
            },
        },
        "VIM": {
            "class": VIM,
            "kwargs": {
                "mnist": dict(princ_dims=0.99),
                "cifar10": dict(princ_dims=40),
                "imagenet": dict(princ_dims=0.99),
            },
            "fit_kwargs": {
                "mnist": dict(feature_layers_id=[-2]),
                "cifar10": dict(feature_layers_id=[-2]),
                "imagenet": dict(feature_layers_id=[-2]),
            },
        },
        "Gram": {
            "class": Gram,
            "kwargs": {
                "mnist": dict(quantile=0.2),
                "cifar10": dict(),
                "imagenet": dict(orders=[1, 2, 3, 4, 5]),
            },
            "fit_kwargs": {
                "mnist": dict(feature_layers_id=["relu1", "relu2"]),
                "cifar10": dict(
                    feature_layers_id=[
                        "layer1.2.conv2",
                        "layer1.2.relu",
                        "layer2.2.conv2",
                        "layer2.2.relu",
                        "layer3.2.conv2",
                        "layer3.2.relu",
                    ]
                ),
                "imagenet": dict(
                    feature_layers_id=[
                        "maxpool",
                        "layer1",
                        "layer2",
                        "layer3",
                        "layer4",
                        "avgpool",
                    ]
                ),
            },
        },
    }

    def __init__(
        self,
        device: torch.device,
        experiments: List[str] = ["mnist", "cifar10"],
        metrics: List[str] = ["auroc", "fpr95tpr", "tpr5fpr"],
    ):
        """Benchmark class.

        Args:
            device (torch.device): device to use.
            experiments (List[str], optional): list of experiments to run.
                Defaults to ["mnist", "cifar10"].
            metrics (List[str], optional): list of metrics to compute.
                Defaults to ["auroc", "fpr95tpr", "tpr5fpr"].
        """
        self.experiments = experiments
        self.metrics = metrics
        self.metrics_dict = {exp: {} for exp in experiments}
        self.perf_dict = {exp: {} for exp in experiments}
        self.device = device

    def _evaluate_one_method(
        self,
        model,
        detector_class,
        detector_kwargs,
        detector_fit_kwargs,
        ds_fit,
        ds_in,
        ds_out_dict,
    ):
        """Evaluate one method and save the time of execution and memory usage.

        Args:
            model (torch.nn.Module): model to evaluate.
            detector_class (oodeel.methods.base.OODBaseDetector): detector class.
            detector_kwargs (dict): detector kwargs.
            detector_fit_kwargs (dict): detector fit kwargs.
            ds_fit (torch.utils.data.DataLoader): in-distribution dataset (fit).
            ds_in (torch.utils.data.DataLoader): in-distribution dataset (test).
            ds_out_dict (dict[str, torch.utils.data.DataLoader]): out-of-distribution
                dataset(s).
        """
        metrics_dict = {}
        # === Fit ===
        with Timer() as t:
            detector = detector_class(**detector_kwargs)
            print("Fitting the detector...")
            detector.fit(model, fit_dataset=ds_fit, verbose=True, **detector_fit_kwargs)
            fit_time = t()
        # === ID scoring ===
        with Timer() as t:
            print("Scoring the detector on ID data...")
            scores_in, _ = detector.score(ds_in, True)
            score_time = t()

        # === OOD scoring ===
        for name, ds_out in ds_out_dict.items():
            print(f"Scoring the detector on OOD data ({name})...")
            scores_out, _ = detector.score(ds_out, True)

            # === metrics ===
            # auroc / fpr95
            print(f"Computing metrics on OOD data ({name})...")
            metrics_dict[name] = bench_metrics(
                (scores_in, scores_out),
                metrics=self.metrics,
            )

        # === Display metrics and time ===
        memory_peak = self._get_reserved_memory_info()

        score_time_per_img = score_time / (len(ds_in.dataset)) * 1000
        perf_stats = {
            "fit_time": fit_time,
            "score_time_per_img": score_time_per_img,
            "memory_peak": memory_peak,
        }

        # print metrics and time
        print("~ Metrics ~")
        for name, metrics in metrics_dict.items():
            print(f"=== Metrics for {name}: ===")
            for k, v in metrics.items():
                print(f"{k:<10} {v:.6f}")
        print("~ Perf stats ~")
        for k, v in perf_stats.items():
            print(f"{k:<10} {v:.6f}")

        return metrics_dict, perf_stats

    def _get_reserved_memory_info(self):
        """Get the peak of reserved memory info since last memory reset.

        Returns:
            tuple: total memory reserved (GB).
        """
        # Calculate peak of GPU memory reserved
        max_memory_reserved = torch.cuda.max_memory_reserved(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        torch.cuda.empty_cache()
        return max_memory_reserved / 1024**3

    def _benchmark_one_experiment(self, experiment: str):
        """Benchmark one experiment.

        Args:
            experiment (str): name of the experiment, e.g. "mnist".
        """
        print(f"\n~~~ BENCHMARK OF {experiment.upper()} ~~~")
        # model
        model = load_model(experiment)
        # datasets
        ds_fit, ds_in, ds_out_dict = load_datasets(experiment, batch_size=60)
        ood_dataset_names = list(ds_out_dict.keys())
        self.metrics_dict[experiment] = {k: {} for k in ood_dataset_names}
        # iterate over methods
        for k, v in self.DETECTORS_CONFIG.items():
            # detector config:
            d_kwargs = v["kwargs"][experiment]
            d_fit_kwargs = v["fit_kwargs"][experiment]
            d_class = v["class"]

            if k == "Gram":
                ds_fit, ds_in, ds_out_dict = load_datasets(experiment, batch_size=32)
            else:
                ds_fit, ds_in, ds_out_dict = load_datasets(experiment, batch_size=60)

            # react detectors
            if (k in self.REACT_DETECTORS) and (experiment not in ["cifar10"]):
                react_quantile = 0.8 if experiment == "mnist" else 0.9
                # for use_react in [True, False]:
                for use_react in [False, True]:
                    k_ = k + ["", " + ReAct"][int(use_react)]
                    d_kwargs.update(
                        dict(
                            use_react=use_react,
                            react_quantile=react_quantile,
                        )
                    )
                    print(f"\n=== {k_} ===")
                    metrics_d, perf_stats = self._evaluate_one_method(
                        model,
                        d_class,
                        d_kwargs,
                        d_fit_kwargs,
                        ds_fit,
                        ds_in,
                        ds_out_dict,
                    )

                    # update metrics
                    for ood_dataset_name in ood_dataset_names:
                        metrics = metrics_d[ood_dataset_name]
                        self.metrics_dict[experiment][ood_dataset_name].update(
                            {k_: metrics}
                        )
                    # update perf stats
                    self.perf_dict[experiment].update({k_: perf_stats})

            # other detectors
            else:
                print(f"\n=== {k} ===")
                # if k == "Gram":
                #     model = load_model(experiment, load_mlp=True)
                metrics_d, perf_stats = self._evaluate_one_method(
                    model, d_class, d_kwargs, d_fit_kwargs, ds_fit, ds_in, ds_out_dict
                )

                # update metrics
                for ood_dataset_name in ood_dataset_names:
                    metrics = metrics_d[ood_dataset_name]
                    self.metrics_dict[experiment][ood_dataset_name].update({k: metrics})

                # update perf stats
                self.perf_dict[experiment].update({k: perf_stats})
                # if k == "Gram":
                #     model = load_model(experiment, load_mlp=False)

    def _get_results_as_dataframe(self, experiment: str):
        """Get results as dataframe for a given experiment.

        Args:
            experiment (str): name of the experiment, e.g. "mnist".
        """
        METRICS_STR = {
            "auroc": "AUROC ($\\uparrow$)",
            "fpr95tpr": "FPR95 ($\\downarrow$)",
            "tpr5fpr": "TPR5 ($\\uparrow$)",
        }
        PERF_STR = {
            "fit_time": "Fit time (s)",
            "score_time_per_img": "Score time (ms / img)",
            "memory_peak": "VRAM usage (GB)",
        }
        df_metrics_dict = {}
        # metrics
        for ood_dataset_name, metrics in self.metrics_dict[experiment].items():
            df_metrics = pd.DataFrame(metrics).T
            df_metrics.columns = [METRICS_STR[m] for m in self.metrics]
            df_metrics.index.name = "Methods"
            df_metrics_dict[ood_dataset_name] = df_metrics

        # perfs
        perfs = self.perf_dict[experiment]
        df_perfs = pd.DataFrame(perfs).T
        df_perfs.columns = [
            PERF_STR[m] for m in ["fit_time", "score_time_per_img", "memory_peak"]
        ]
        df_perfs.index.name = "Methods"
        return df_metrics_dict, df_perfs

    def export_results(self, dir_path: str = "."):
        """Export results as markdown and latex tables for all experiments.

        Args:
            dir_path (str, optional): directory where to save the results.
                Defaults to '.'.
        """
        os.makedirs(dir_path, exist_ok=True)
        for experiment in self.experiments:
            df_metrics_dict, df_perfs = self._get_results_as_dataframe(experiment)
            for ood_dataset_name, df_metrics in df_metrics_dict.items():
                # merge metrics and time
                df = pd.concat([df_metrics, df_perfs], axis=1)
                # export as markdown
                fp_markdown = os.path.join(
                    dir_path, f"torch_results_{experiment}_vs_{ood_dataset_name}.md"
                )
                with open(fp_markdown, "w") as f:
                    df.to_markdown(f, floatfmt=".3f")
                # export as latex with caption
                fp_latex = os.path.join(
                    dir_path, f"torch_results_{experiment}_vs_{ood_dataset_name}.tex"
                )
                caption = f"Results on {experiment}."
                with open(fp_latex, "w") as f:
                    df.to_latex(f, caption=caption, float_format="%.3f")
                # export plots
                self.plot_results(experiment, dir_path=dir_path, save_plot_as_pdf=True)

    def plot_results(
        self, experiment: str, dir_path: str = ".", save_plot_as_pdf: bool = False
    ):
        """Plot results as bar plot for a given experiment. Each method is referenced
        in the legend of the plot and the bar plots are grouped by metrics.

        Args:
            experiment (str): name of the experiment, e.g. "mnist".
            dir_path (str, optional): directory where to save the plot. Defaults to ".".
            save_plot_as_pdf (bool, optional): whether to save the plot as pdf.
                Defaults to False.
        """
        sns.set_theme(style="darkgrid")
        sns.set_context("paper")
        df_metrics_dict, df_perfs = self._get_results_as_dataframe(experiment)
        # sns.set_palette(sns.color_palette("colorblind", n_colors=len(df_metrics)))

        for ood_dataset_name, df_metrics in df_metrics_dict.items():
            # === plot metrics ===
            ax = df_metrics.T.plot.bar(figsize=(9, 3.1))
            ax.set_ylim(0, 1)
            ax.set_xlabel("Metrics")
            ax.set_ylabel("Scores")
            plt.xticks(rotation=0)
            plt.tight_layout()
            plt.legend(ncol=3, fontsize="x-small")
            # save plot
            if save_plot_as_pdf:
                fp_plot = os.path.join(
                    dir_path, f"torch_metrics_{experiment}_vs_{ood_dataset_name}.pdf"
                )
                plt.savefig(fp_plot)
            plt.close()

        # === plot perfs ===
        ax = df_perfs.T.iloc[:2].T.plot.barh(
            figsize=(9, 3.1),
            subplots=True,
            sharex=False,
            sharey=True,
            layout=(1, 2),
            legend=False,
        )
        plt.tight_layout()
        # save plot
        if save_plot_as_pdf:
            fp_plot = os.path.join(dir_path, f"torch_perfs_{experiment}.pdf")
            plt.savefig(fp_plot)

    def run(self):
        for experiment in self.experiments:
            self._benchmark_one_experiment(experiment)


if __name__ == "__main__":
    dir_path = os.path.expanduser("~/") + "1_OODEEL/experiments/benchmark/results/torch"
    os.makedirs(dir_path, exist_ok=True)

    # run benchmark
    # benchmark = BenchmarkTorch(device, experiments=["cifar10", "mnist"])
    benchmark = BenchmarkTorch(device, experiments=["imagenet"])
    benchmark.run()
    benchmark.export_results(dir_path=dir_path)
    print("Results exported in:", dir_path)
