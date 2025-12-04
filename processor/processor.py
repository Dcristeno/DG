'''
Reference: NCR (NeurIPS 2021)
Modified for Weights & Biases logging by 光执
'''
import logging
import os
import time
import torch
from utils.meter import AverageMeter
from utils.metrics import Evaluator
from utils.comm import get_rank, synchronize
# from torch.utils.tensorboard import SummaryWriter  # disabled in favor of W&B
from prettytable import PrettyTable
import numpy as np
from matplotlib import pyplot as plt
from pytorch_lightning.utilities.memory import garbage_collection_cuda
from pylab import xticks, yticks, np
from sklearn.metrics import confusion_matrix
from sklearn.mixture import GaussianMixture

################### CODE FOR THE BETA MODEL  ########################

import scipy.stats as stats
import wandb  # ✅ ADDED: W&B import

def weighted_mean(x, w):
    return np.sum(w * x) / np.sum(w)

def fit_beta_weighted(x, w):
    x_bar = weighted_mean(x, w)
    s2 = weighted_mean((x - x_bar)**2, w)
    alpha = x_bar * ((x_bar * (1 - x_bar)) / s2 - 1)
    beta = alpha * (1 - x_bar) / x_bar
    return alpha, beta

class BetaMixture1D(object):
    def __init__(self, max_iters=10,
                 alphas_init=[1, 2],
                 betas_init=[2, 1],
                 weights_init=[0.5, 0.5]):
        self.alphas = np.array(alphas_init, dtype=np.float64)
        self.betas = np.array(betas_init, dtype=np.float64)
        self.weight = np.array(weights_init, dtype=np.float64)
        self.max_iters = max_iters
        self.lookup = np.zeros(100, dtype=np.float64)
        self.lookup_resolution = 100
        self.lookup_loss = np.zeros(100, dtype=np.float64)
        self.eps_nan = 1e-12

    def likelihood(self, x, y):
        return stats.beta.pdf(x, self.alphas[y], self.betas[y])

    def weighted_likelihood(self, x, y):
        return self.weight[y] * self.likelihood(x, y)

    def probability(self, x):
        return sum(self.weighted_likelihood(x, y) for y in range(2))

    def posterior(self, x, y):
        return self.weighted_likelihood(x, y) / (self.probability(x) + self.eps_nan)

    def responsibilities(self, x):
        r = np.array([self.weighted_likelihood(x, i) for i in range(2)])
        r[r <= self.eps_nan] = self.eps_nan
        r /= r.sum(axis=0)
        return r

    def score_samples(self, x):
        return -np.log(self.probability(x))

    def fit(self, x):
        x = np.copy(x)
        eps = 1e-4
        x[x >= 1 - eps] = 1 - eps
        x[x <= eps] = eps

        for i in range(self.max_iters):
            r = self.responsibilities(x)
            self.alphas[0], self.betas[0] = fit_beta_weighted(x, r[0])
            self.alphas[1], self.betas[1] = fit_beta_weighted(x, r[1])
            self.weight = r.sum(axis=1)
            self.weight /= self.weight.sum()
        return self

    def predict(self, x):
        return self.posterior(x, 1) > 0.5

    def create_lookup(self, y):
        x_l = np.linspace(0+self.eps_nan, 1-self.eps_nan, self.lookup_resolution)
        lookup_t = self.posterior(x_l, y)
        lookup_t[np.argmax(lookup_t):] = lookup_t.max()
        self.lookup = lookup_t
        self.lookup_loss = x_l

    def look_lookup(self, x):
        x_i = x.clone().cpu().numpy()
        x_i = np.array((self.lookup_resolution * x_i).astype(int))
        x_i[x_i < 0] = 0
        x_i[x_i == self.lookup_resolution] = self.lookup_resolution - 1
        return self.lookup[x_i]

    def __str__(self):
        return 'BetaMixture1D(w={}, a={}, b={})'.format(self.weight, self.alphas, self.betas)


def split_prob(prob, threshld):
    if prob.min() > threshld:
        print('No estimated noisy data. Enforce the 1/100 data with small probability to be unlabeled.')
        threshld = np.sort(prob)[len(prob)//100]
    pred = (prob > threshld)
    return (pred + 0)


def get_loss(model, data_loader, args):
    """Modified to return both binary prediction AND raw clean probability"""
    logger = logging.getLogger("DG.train")
    model.eval()
    device = "cuda"
    data_size = data_loader.dataset.__len__()
    real_labels = data_loader.dataset.real_correspondences
    c = args.prob
    d = 1 - args.prob
    lossA, lossB, simsA, simsB = torch.zeros(data_size), torch.zeros(data_size), torch.zeros(data_size), torch.zeros(data_size)

    for i, batch in enumerate(data_loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        index = batch['index']
        with torch.no_grad():
            la, lb, sa, sb = model.compute_per_loss(batch, args)
            for b in range(la.size(0)):
                lossA[index[b]] = la[b]
                lossB[index[b]] = lb[b]
                simsA[index[b]] = sa[b]
                simsB[index[b]] = sb[b]
            if i == 0:
                logger.info(f'get_loss: processing batch {i}')

    losses_A = (lossA - lossA.min()) / (lossA.max() - lossA.min() + 1e-8)
    losses_B = (lossB - lossB.min()) / (lossB.max() - lossB.min() + 1e-8)
    input_loss_A = losses_A.reshape(-1, 1)
    input_loss_B = losses_B.reshape(-1, 1)

    logger.info(f'Fitting GMM on losses (A, B)')

    if model.args.noisy_rate > 0.4 or model.args.dataset_name == 'RSTPReid':
        gmm_A = GaussianMixture(n_components=2, max_iter=100, tol=1e-4, reg_covar=1e-6)
        gmm_B = GaussianMixture(n_components=2, max_iter=100, tol=1e-4, reg_covar=1e-6)
    else:
        gmm_A = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
        gmm_B = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)

    gmm_A.fit(input_loss_A.cpu().numpy())
    prob_A = gmm_A.predict_proba(input_loss_A.cpu().numpy())
    prob_A = prob_A[:, gmm_A.means_.argmin()]  # clean prob (smaller mean = smaller loss = cleaner)
    pred_A = split_prob(prob_A, c)

    gmm_B.fit(input_loss_B.cpu().numpy())
    prob_B = gmm_B.predict_proba(input_loss_B.cpu().numpy())
    prob_B = prob_B[:, gmm_B.means_.argmin()]
    pred_B = split_prob(prob_B, d)

    # ✅ Return: binary preds + raw clean probabilities (for histogram)
    return (
        torch.Tensor(pred_A), torch.Tensor(pred_B),
        torch.Tensor(prob_A), torch.Tensor(prob_B)
    )


def do_train(start_epoch, args, model, train_loader, evaluator, optimizer,
             scheduler, checkpointer):

    log_period = args.log_period
    eval_period = args.eval_period
    device = "cuda"
    num_epoch = args.num_epoch
    arguments = {}
    arguments["num_epoch"] = num_epoch
    arguments["iteration"] = 0

    logger = logging.getLogger("DG.train")
    logger.info('start training')

    # ✅ Initialize W&B (only master process)
    if get_rank() == 0:
        wandb.init(
            project="NCR-noisy-correspondence",
            name=f"{args.dataset_name}_noise{args.noisy_rate}_prob{args.prob}",
            config=vars(args),
            dir=args.output_dir,
            # group=f"{args.dataset_name}_noise{args.noisy_rate}",
            # tags=["NCR", "consensus", "GMM"]
        )
        logger.info(f"W&B run initialized: {wandb.run.name} | URL: {wandb.run.url}")

    meters = {
        "loss": AverageMeter(),
        "bge_loss": AverageMeter(),
        "tse_loss": AverageMeter(),
        "id_loss": AverageMeter(),
        "img_acc": AverageMeter(),
        "txt_acc": AverageMeter(),
    }

    best_top1 = 0.0
    sims = []

    for epoch in range(start_epoch, num_epoch + 1):
        start_time = time.time()
        for meter in meters.values():
            meter.reset()
        garbage_collection_cuda()
        model.epoch = epoch

        # Get consensus labels
        pred_A, pred_B, prob_A_raw, prob_B_raw = get_loss(model, train_loader, args)
        consensus_division = pred_A + pred_B
        consensus_division[consensus_division == 1] += torch.randint(0, 2, size=(((consensus_division == 1) + 0).sum(),))
        label_hat = consensus_division.clone()
        label_hat[consensus_division > 1] = 1
        label_hat[consensus_division <= 1] = 0

        model.train()
        for n_iter, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            index = batch['index']
            batch['label_hat'] = label_hat[index.cpu()]

            ret = model(batch)
            total_loss = sum([v for k, v in ret.items() if "loss" in k])
            batch_size = batch['images'].shape[0]

            meters['loss'].update(total_loss.item(), batch_size)
            meters['bge_loss'].update(ret.get('bge_loss', 0), batch_size)
            meters['tse_loss'].update(ret.get('tse_loss', 0), batch_size)
            if 'img_acc' in ret:
                meters['img_acc'].update(ret['img_acc'], batch_size)
            if 'txt_acc' in ret:
                meters['txt_acc'].update(ret['txt_acc'], batch_size)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            synchronize()

            if (n_iter + 1) % log_period == 0:
                info_str = f"Epoch[{epoch}] Iteration[{n_iter + 1}/{len(train_loader)}]"
                for k, v in meters.items():
                    if v.count > 0:
                        info_str += f", {k}: {v.avg:.4f}"
                info_str += f", Base Lr: {scheduler.get_lr()[0]:.2e}"
                logger.info(info_str)

                # ✅ W&B: log intermediate train metrics (per log_period)
                if get_rank() == 0:
                    wandb.log({
                        "train/iter_loss": meters['loss'].val,
                        "train/iter_lr": scheduler.get_lr()[0]
                    }, step=arguments["iteration"])
                arguments["iteration"] += 1

        scheduler.step()

        # ✅ W&B: log epoch-level train metrics
        if get_rank() == 0:
            end_time = time.time()
            time_per_batch = (end_time - start_time) / (n_iter + 1)
            wandb_log_dict = {
                "epoch": epoch,
                "train/epoch_time": end_time - start_time,
                "train/time_per_batch": time_per_batch,
                "lr": scheduler.get_lr()[0],
                "train/loss": meters['loss'].avg,
                "train/bge_loss": meters['bge_loss'].avg,
                "train/tse_loss": meters['tse_loss'].avg,
            }
            if meters['img_acc'].count > 0:
                wandb_log_dict["train/img_acc"] = meters['img_acc'].avg
            if meters['txt_acc'].count > 0:
                wandb_log_dict["train/txt_acc"] = meters['txt_acc'].avg
            wandb.log(wandb_log_dict, step=epoch)

        # Validation
        if epoch % eval_period == 0:
            if get_rank() == 0:
                logger.info("Validation Results - Epoch: {}".format(epoch))
                with torch.no_grad():
                    if args.distributed:
                        top1 = evaluator.eval(model.module.eval())
                    else:
                        top1 = evaluator.eval(model.eval())
                torch.cuda.empty_cache()

                # ✅ W&B: log validation metric
                wandb.log({"val/R1": top1}, step=epoch)

                # ✅ W&B: log noise estimation stats (CRITICAL for NCR analysis)
                clean_ratio_A = pred_A.float().mean().item()
                clean_ratio_B = pred_B.float().mean().item()
                consensus_clean = label_hat.float().mean().item()

                wandb.log({
                    "train/clean_ratio_A": clean_ratio_A,
                    "train/clean_ratio_B": clean_ratio_B,
                    "train/consensus_clean_ratio": consensus_clean,
                }, step=epoch)

                # ✅ W&B: upload raw clean probability histograms (shows bimodality!)
                wandb.log({
                    "train/prob_A_hist": wandb.Histogram(prob_A_raw.numpy()),
                    "train/prob_B_hist": wandb.Histogram(prob_B_raw.numpy()),
                }, step=epoch)

                # Save best model & W&B artifact
                if best_top1 < top1:
                    best_top1 = top1
                    arguments["epoch"] = epoch
                    ckpt_path = checkpointer.save("best", **arguments)

                    # ✅ W&B artifact: upload best model
                    artifact = wandb.Artifact(
                        name=f"model-{args.dataset_name}",
                        type="model",
                        metadata={
                            "R1": top1,
                            "epoch": epoch,
                            "noisy_rate": args.noisy_rate,
                            "prob": args.prob,
                            "clean_ratio_consensus": consensus_clean,
                        }
                    )
                    if os.path.exists(ckpt_path):
                        artifact.add_file(ckpt_path)
                        wandb.log_artifact(artifact)
                        logger.info(f"Best model artifact logged to W&B: {ckpt_path}")

    if get_rank() == 0:
        logger.info(f"Training finished. Best R1: {best_top1} at epoch {arguments.get('epoch', -1)}")
        wandb.finish()


def do_inference(model, test_img_loader, test_txt_loader):
    logger = logging.getLogger("DG.test")
    logger.info("Enter inferencing")
    evaluator = Evaluator(test_img_loader, test_txt_loader)
    top1 = evaluator.eval(model.eval())
    if get_rank() == 0:
        logger.info(f"Test R1: {top1}")
        wandb.init(project="NCR-inference", name=f"test_{top1:.2f}", reinit=True)
        wandb.log({"test/R1": top1})
        wandb.finish()