"""
implement on plotting callback that inputs 2 generators,
performs inference on each signal in the val and train set
and plots the results and saves the image

"""
import os
import json

import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt
from keras.callbacks import Callback
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, matthews_corrcoef, recall_score, precision_score


class ClassificationCallback(Callback):

    def __init__(self, train_generator, val_generator, summary_path, loss, model=None):
        Callback.__init__(self)
        self.train_generator = train_generator
        self.val_generator = val_generator
        self.summary_path = summary_path
        self.loss = loss

    def get_metrics(self, y_true, y_pred, thresh=0.45):
        loss = self.loss(y_true, y_pred)
        y_pred[y_pred < thresh] = 0
        y_pred[y_pred > 0] = 1
        conf_mat = confusion_matrix(y_true, y_pred).astype(int).tolist()
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).astype(int).ravel()
        try:
            specificity = tn / (tn + fp)
        except ZeroDivisionError as e:
            specificity = 0
        matthews = matthews_corrcoef(y_true, y_pred)
        recall = recall_score(y_true, y_pred, zero_division=0)
        precision = precision_score(y_true, y_pred, zero_division=0)
        return loss, conf_mat, matthews, recall, precision, specificity

    def on_epoch_end(self, epoch, logs=None):
        train_labels, train_predictions = self.run(self.train_generator)
        val_labels, val_predictions = self.run(self.val_generator)
        train_loss, train_conf_mat, train_matthews, train_recall, train_precision, train_specificity = self.get_metrics(
            train_labels, train_predictions)
        val_loss, val_conf_mat, val_matthews, val_recall, val_precision, val_specificity = self.get_metrics(val_labels,
                                                                                                            val_predictions)

        update = {
            'train_loss': train_loss.astype(float),
            'train_conf_mat': train_conf_mat,
            'train_matthews': train_matthews,
            'train_precision': train_precision,
            'train_recall': train_recall,
            'train_specificity': train_specificity,
            'val_loss': val_loss.astype(float),
            'val_conf_mat': val_conf_mat,
            'val_matthews': val_matthews,
            'val_precision': val_precision,
            'val_recall': val_recall,
            'val_specificity': val_specificity,
        }
        self.write_summary(epoch + 1, update)
        self.write_graph()

    def on_train_begin(self, logs=None):
        """
        if summary_path_dir doesnt exist create dir call write_sumamry
        """
        self.graph_path = self.summary_path.parents[0].joinpath('graphs').resolve()
        if not self.graph_path.exists():
            self.graph_path.mkdir(parents=True)
        if not self.summary_path.exists():
            self.save_summary({})

    @classmethod
    def load_summary(cls, summary_path):
        with open(summary_path, 'r') as f:
            s = json.load(f)
        return s

    def save_summary(self, summary):
        with open(self.summary_path, 'w') as f:
            json.dump(summary, f, indent=4, sort_keys=True)

    def write_summary(self, key, update):
        summary = self.load_summary(self.summary_path)
        summary.update({f'{key:02d}': update})
        self.save_summary(summary)

    @classmethod
    def get_gridspec(cls):
        fig10 = plt.figure(constrained_layout=True)
        gs0 = fig10.add_gridspec(1, 2)

        loss_gs = gs0[0].subgridspec(1, 1)
        metric_gs = gs0[1].subgridspec(2, 2)
        fig10.add_subplot(loss_gs[0])

        for dataset in range(2):
            for irow_metric in range(2):
                for jrow_metric in range(2):
                    fig10.add_subplot(metric_gs[irow_metric, jrow_metric])
        return fig10.axes

    def write_graph(self, train_color='r', val_color='g'):
        loss_axes, recall_axes, precision_axes, matthews_axes, specificity_axes = self.get_gridspec()
        summary = self.load_summary(self.summary_path)
        epochs = []
        train_loss = []
        val_loss = []
        t_recall = []
        t_precision = []
        t_matthews = []
        t_specificity = []
        v_recall = []
        v_precision = []
        v_matthews = []
        v_specificity = []
        for epoch, v in summary.items():
            epochs.append(int(epoch))
            train_loss.append(v['train_loss'])
            val_loss.append(v['val_loss'])
            t_recall.append(v['train_recall'])
            t_precision.append(v['train_precision'])
            t_matthews.append(v['train_matthews'])
            t_specificity.append(v['train_specificity'])
            v_recall.append(v['val_recall'])
            v_precision.append(v['val_precision'])
            v_matthews.append(v['val_matthews'])
            v_specificity.append(v['val_specificity'])


        loss_axes.plot(epochs, train_loss, train_color, label='train_loss')
        loss_axes.plot(epochs, val_loss, val_color, label='val_loss')

        recall_axes.plot(epochs, t_recall, train_color, label='train_recall')
        precision_axes.plot(epochs, t_precision, train_color, label='train_precision')
        matthews_axes.plot(epochs, t_matthews, train_color, label='train_matthews')
        specificity_axes.plot(epochs, t_specificity, train_color, label='train_specificity')

        recall_axes.plot(epochs, v_recall, val_color, label='val_recall')
        precision_axes.plot(epochs, v_precision, val_color, label='val_precision')
        matthews_axes.plot(epochs, v_matthews, val_color, label='val_matthews')
        specificity_axes.plot(epochs, v_specificity, val_color, label='val_specificity')
        plt.legend()
        plt.savefig(self.graph_path.joinpath('metrics.png').resolve())
        plt.close()
        #plt.show()


    def run(self, gen_obj):
        y_true = np.zeros(gen_obj.steps * gen_obj.batch_size)
        y_pred = np.zeros(gen_obj.steps * gen_obj.batch_size)
        for i, (data, labels) in tqdm(enumerate(gen_obj.corruption_val_generator())):
            if i == gen_obj.steps:
                break
            start_index = i * gen_obj.batch_size
            end_index = start_index + gen_obj.batch_size
            pred = self.model.predict_on_batch(data)
            y_true[start_index:end_index] = labels
            y_pred[start_index:end_index] = pred.reshape(-1)
        return y_true, y_pred

    @classmethod
    def plot_summary(cls, summary_path):
        graph_path = summary_path.parent[0].joinpath('graphs').resole()
        graph_path.makedir()
        summary = cls.load_summary(summary_path)
        train_labels = summary['train_labels']
        val_labels = summary['val_labels']
        train_inds = train_labels.argsort()
        val_inds = val_labels.argsort()
        train_labels = np.array(train_labels[train_inds])
        val_labels = np.array(val_labels[val_inds])
        del summary['train_labels'], summary['val_labels']
        f, (ax1, ax2) = plt.subplots(1, 2)
        for k, v in summary.items():
            train_pred = np.array(v['train_prediction'])[train_inds]
            val_pred = np.array(v['val_prediction'])[val_inds]
            ax1.plot(train_labels, 'g')
            ax1.plot(train_pred, 'r')
            ax2.plot(val_labels, 'g')
            ax2.plot(val_pred, 'r')
            plt.savefig(graph_path.joinpath(f'epoch_{k}.png'))
            ax1.cla()
            ax2.cla()


if __name__ == '__main__':
    from image_builder import EmgImageGenerator
    from pathlib import Path
    from datetime import datetime

    a = ClassificationCallback.get_gridspec()
    plt.show()
    from sys import exit
    exit()
    train_path = Path(__file__, '..', 'files', 'dl_train_annotations.csv')
    val_path = Path(__file__, '..', 'files', 'dl_val_annotations.csv')
    date_id = datetime.now().strftime('%Y%m%d%H%M')

    experiment_dir = Path(__file__, '..', 'files', 'deep_learning', date_id).resolve()

    summary_path = experiment_dir.joinpath('summaries', 'summary.json')
    train_gen = EmgImageGenerator(train_path, 16)
    val_gen = EmgImageGenerator(val_path, 16)
    loss = lambda y, p: ((y - p) ** 2).mean()

    p = ClassificationCallback(train_gen, val_gen, summary_path, loss)
    p.write_graph(1)
    #p.on_train_begin()
    #p.on_epoch_end(1)
