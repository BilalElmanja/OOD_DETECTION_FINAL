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
import sys
sys.path.append("../")
from oodeel.methods import MLS, Energy, Entropy, DKNN, Gram, Mahalanobis, ODIN, VIM
from methods import  K_Means, PCA_KNN, NMF_KNN, PCA_MAHALANOBIS, NMF_MAHALANOBIS, PCA_unique_class_KNN, PCA_Unique_Class_Mahalanobis, NMF_Unique_Classes_KNN, NMF_Unique_Class_Mahalanobis
from data_preprocessing import get_train_dataset_cifar10, get_test_dataset_cifar10, get_train_dataset_cifar100, get_test_dataset_cifar100, get_test_dataset_places365, get_test_dataset_svhn, get_test_dataset_texture, get_test_dataset_Tiny, get_test_dataset_NINCO, get_test_dataset_OpenImage_O, get_train_dataset_inaturalist, get_test_dataset_SSB_hard
from models import load_pretrained_weights_32
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



# models
def load_model(experiment="cifar10", load_mlp=False):
    
    # model trained on CIFAR10
    if experiment == "cifar10":
        model = load_pretrained_weights_32()

    # model trained on CIFAR10
    elif experiment == "cifar100":
        model = load_pretrained_weights_32(dataset='CIFAR-100', model_version='s0', num_classes=100)

    else:
        raise ValueError("`experiment` should be 'cifar10' or 'cifar100'.")
    
    model.to(device)
    model.eval()
    return model


# datasets
def load_datasets(experiment: str = "cifar100", batch_size: int = 128):

    # CIFAR10 
    if experiment == "cifar10":
        # 1a- load in-distribution dataset: CIFAR-10
        ds_fit = get_train_dataset_cifar10()
        ds_in = get_test_dataset_cifar10()
        ds_out_dict = {
            "cifar100": get_test_dataset_cifar100(),
            "svhn" : get_test_dataset_svhn(),
            "places365" : get_test_dataset_places365(),
            "texture" : get_test_dataset_texture(),
            "Tin": get_test_dataset_Tiny(),
        }

     # CIFAR100
    elif experiment == "cifar100":
        # 1a- load in-distribution dataset: CIFAR-10
        ds_fit = get_train_dataset_cifar100()
        ds_in = get_test_dataset_cifar100()
        ds_out_dict = {
            "cifar10": get_test_dataset_cifar10(),
            "svhn" : get_test_dataset_svhn(),
            "places365" : get_test_dataset_places365(),
            "texture" : get_test_dataset_texture(),
            "Tin": get_test_dataset_Tiny(),
        }
 

    else:
        raise ValueError("`experiment` should be or 'cifar10' or cifar100")

    print("moving data to : ", device)
    for x, y in ds_fit:
        x.to(device)
        y.to(device)

    for x, y in ds_in:
        x.to(device)
        y.to(device)

    for name, ds_out in ds_out_dict.items():
        for x, y in ds_out:
            x.to(device)
            y.to(device)
       
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
                "cifar10": dict(),
                "cifar100":dict(),
            },
            "fit_kwargs": {
                "cifar10": dict(),
                "cifar100":dict(),
            },
        },
        "MSP": {
            "class": MLS,
            "kwargs": {
                "cifar10": dict(output_activation="softmax"),
                "cifar100":dict(output_activation="softmax"),
            },
            "fit_kwargs": {
                "cifar10": dict(),
                "cifar100":dict(),
            },
        },
        "Energy": {
            "class": Energy,
            "kwargs": {
                "cifar10": dict(),
                "cifar100":dict(),
            },
            "fit_kwargs": {
                "cifar10": dict(),
                "cifar100":dict(),
            },
        },
        "Entropy": {
            "class": Entropy,
            "kwargs": {
                "cifar10": dict(),
                "cifar100":dict(),
            },
            "fit_kwargs": {
                "cifar10": dict(),
                "cifar100":dict(),
            },
        },
        "ODIN": {
            "class": ODIN,
            "kwargs": {
                "cifar10": dict(temperature=1000),
                "cifar100":dict(temperature=1000),
            },
            "fit_kwargs": {
                "cifar10": dict(),
                "cifar100":dict(),
            },
        },
        "DKNN": {
            "class": DKNN,
            "kwargs": {
                "cifar10": dict(nearest=50),
                "cifar100":dict(nearest=50),
            },
            "fit_kwargs": {
                "cifar10": dict(feature_layers_id=[-1]),
                "cifar100": dict(feature_layers_id=[-1]),
            },
        },
        "Mahalanobis": {
            "class": Mahalanobis,
            "kwargs": {
                "cifar10": dict(),
                "cifar100":dict(),
            },
            "fit_kwargs": {
                "cifar10": dict(feature_layers_id=[-1]),
                "cifar100": dict(feature_layers_id=[-1]),
            },
        },
        # "VIM": {
        #     "class": VIM,
        #     "kwargs": {
        #         "mnist": dict(princ_dims=0.99),
        #         "cifar10": dict(princ_dims=40),
        #         "imagenet": dict(princ_dims=0.99),
        #     },
        #     "fit_kwargs": {
        #         "mnist": dict(feature_layers_id=[-1]),
        #         "cifar10": dict(feature_layers_id=[-1]),
        #         "imagenet": dict(feature_layers_id=[-1]),
        #     },
        # },
        # "Gram": {
        #     "class": Gram,
        #     "kwargs": {
        #         "mnist": dict(quantile=0.2),
        #         "cifar10": dict(),
        #         "imagenet": dict(orders=[1, 2, 3, 4, 5]),
        #     },
        #     "fit_kwargs": {
        #         "mnist": dict(feature_layers_id=["relu1", "relu2"]),
        #         "cifar10": dict(
        #             feature_layers_id=[
        #                 "layer1.2.conv2",
        #                 "layer1.2.relu",
        #                 "layer2.2.conv2",
        #                 "layer2.2.relu",
        #                 "layer3.2.conv2",
        #                 "layer3.2.relu",
        #             ]
        #         ),
        #         "imagenet": dict(
        #             feature_layers_id=[
        #                 "maxpool",
        #                 "layer1",
        #                 "layer2",
        #                 "layer3",
        #                 "layer4",
        #                 "avgpool",
        #             ]
        #         ),
        #     },
        # },
        "Kmeans": {
            "class": K_Means,
            "kwargs": {
                "cifar10": dict(),
                "cifar100":dict(),
            },
            "fit_kwargs": {
                "cifar10": dict(feature_layers_id=[-1]),
                "cifar100": dict(feature_layers_id=[-1]),
            },
        },
        "PCA_KNN": {
            "class": PCA_KNN,
            "kwargs": {
                "cifar10": dict(),
                "cifar100":dict(),
            },
            "fit_kwargs": {
                "cifar10": dict(feature_layers_id=[-1]),
                "cifar100": dict(feature_layers_id=[-1]),
            },
        },
        "PCA_Mahalanobis": {
            "class": PCA_MAHALANOBIS,
            "kwargs": {
                "cifar10": dict(),
                "cifar100":dict(),
            },
            "fit_kwargs": {
                "cifar10": dict(feature_layers_id=[-1]),
                "cifar100": dict(feature_layers_id=[-1]),
            },
        },
        # "NMF_KNN": {
        #     "class": NMF_KNN,
        #     "kwargs": {
        #         "cifar10": dict(),
        #         "cifar100":dict(),
        #     },
        #     "fit_kwargs": {
        #         "cifar10": dict(feature_layers_id=[-1]),
        #         "cifar100": dict(feature_layers_id=[-1]),
        #     },
        # },
        # "NMF_Mahalanobis": {
        #     "class": NMF_MAHALANOBIS,
        #     "kwargs": {
        #         "cifar10": dict(),
        #         "cifar100": dict(),
        #     },
        #     "fit_kwargs": {
        #         "cifar10": dict(feature_layers_id=[-1]),
        #         "cifar100": dict(feature_layers_id=[-1]),
        #     },
        # },
        "PCA_per_class_knn": {
            "class": PCA_unique_class_KNN,
            "kwargs": {
                "cifar10": dict(),
                "cifar100": dict(),
            },
            "fit_kwargs": {
                "cifar10": dict(feature_layers_id=[-1]),
                "cifar100": dict(feature_layers_id=[-1]),
            },
        },
        "pca_per_class_mahalanobis": {
            "class": PCA_Unique_Class_Mahalanobis,
            "kwargs": {
                "cifar10": dict(),
                "cifar100": dict(),
            },
            "fit_kwargs": {
                "cifar10": dict(feature_layers_id=[-1]),
                "cifar100": dict(feature_layers_id=[-1]),
            },
        },
        # "NMF_per_class": {
        #     "class": NMF_Unique_Classes_KNN,
        #     "kwargs": {
        #         "cifar10": dict(),
        #         "cifar100": dict(),
        #     },
        #     "fit_kwargs": {
        #         "cifar10": dict(feature_layers_id=[-1]),
        #         "cifar100": dict(feature_layers_id=[-1]),
        #     },
        # },
        # "NMF_per_class_mahalanobis": {
        #     "class": NMF_Unique_Class_Mahalanobis,
        #     "kwargs": {
        #         "cifar10": dict(),
        #         "cifar100": dict(),
        #     },
        #     "fit_kwargs": {
        #         "cifar10": dict(feature_layers_id=[-1]),
        #         "cifar100": dict(feature_layers_id=[-1]),
        #     },
        # },
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
            detector.fit(model, fit_dataset=ds_fit,  **detector_fit_kwargs)
            fit_time = t()
        # === ID scoring ===
        with Timer() as t:
            print("Scoring the detector on ID data...")
            scores_in, _ = detector.score(ds_in)
            score_time = t()

        # === OOD scoring ===
        for name, ds_out in ds_out_dict.items():
            print(f"Scoring the detector on OOD data ({name})...")
            scores_out, _ = detector.score(ds_out)

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
            if (k in self.REACT_DETECTORS) and (experiment not in ["cifar10", "cifar100"]):
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
    dir_path = os.path.expanduser("~/") + "./results/cifar100_layer_1"
    os.makedirs(dir_path, exist_ok=True)

    # run benchmark
    benchmark = BenchmarkTorch(device, experiments=["cifar100"])
    # benchmark = BenchmarkTorch(device, experiments=["imagenet"])
    benchmark.run()
    benchmark.export_results(dir_path=dir_path)
    print("Results exported in:", dir_path)