import ROOT as pyr
import numpy as np
import pandas as pd
import root_numpy
import matplotlib.pyplot as plt
import math
import keras

def ROC_graph(y_predict, y_test, weight=None, points=50):
    y_predict_sig_weight = []
    y_predict_bg_weight = []
    if weight is None:
        for i in range(len(y_predict)):
            if y_test[i] == 1:
                y_predict_sig_weight.append((y_predict[i], 1))
            else:
                y_predict_bg_weight.append((y_predict[i], 1))
    else:
        for i in range(len(y_predict)):
            if type(weight) is not list:
                if y_test[i] == 1:
                    y_predict_sig_weight.append((y_predict[i], weight.ravel()[i]))
                else:
                    y_predict_bg_weight.append((y_predict[i], weight.ravel()[i]))
            else:
                if y_test[i] == 1:
                    y_predict_sig_weight.append((y_predict[i], weight[i]))
                else:
                    y_predict_bg_weight.append((y_predict[i], weight[i]))
    
    y_predict_sig_weight.sort(key = lambda x: x[0])
    y_predict_bg_weight.sort(key = lambda x: x[0])

    graph = pyr.TGraph(points)
    graph.SetPoint(points-1, 0, 0)
    for i in range(1, points):
        graph.SetPoint(points-1-i, i/points, signal_acc(y_predict_bg_weight, y_predict_sig_weight, i/points))
        #print((i/20., signal_acc(y_predict_bg_weight, y_predict_sig_weight, i/20.)))
    
    return graph

def calculate_AUC(y_predict, y_test, weight=None):
    graph = ROC_graph(y_predict, y_test, weight)
    return graph.Integral(0, -1)+0.5

def to_pandas(root_filename, tree_name, selection, use_columns, use_columns_name, add_weight=True, isSignal=True):
    if add_weight:
        use_columns_ = use_columns + ["Max$(SFlepton*SFPU*csvrsw[0]*ttxrew*toprew*GenWeight*SFtrig)"]
        use_columns_name_ = use_columns_name + ["weight"]
    else:
        use_columns_ = use_columns
        use_columns_name_ = use_columns_name
    X = root_numpy.root2array(root_filename, tree_name, selection=selection, branches=use_columns_, step=1)
    df = pd.DataFrame(data=X)
    df.columns = use_columns_name_
    df = df.assign(target=pd.Series(np.asarray([1]*len(X)) if isSignal else np.asarray([0]*len(X))))

    return df

def normalize_columns(data, lower, upper):
    rng = np.asarray(upper) - np.asarray(lower)
    normalised = (data - np.asarray(lower))/rng
    return normalised

def plot_output_dist_channels(model, variables_used, data, channels, weightset = None, wantSignal = False, custom_suffix = ""):
    for jets_, mtags_ in channels:
        if '+' in str(jets_) and '+' in str(mtags_):
            jets = int(jets_[:-1])
            mtags = int(mtags_[:-1])
            data_filtered = data.loc[(data["nJets"] >= jets) & (data["nMtags"] >= mtags) & (data["target"] == 0)]
        elif '+' in str(mtags_):
            jets = int(jets_)
            mtags = int(mtags_[:-1])
            data_filtered = data.loc[(data["nJets"] == jets) & (data["nMtags"] >= mtags) & (data["target"] == 0)]
        elif '+' in str(jets_):
            jets = int(jets_[:-1])
            mtags = int(mtags_)
            data_filtered = data.loc[(data["nJets"] >= jets) & (data["nMtags"] == mtags) & (data["target"] == 0)]
        else:
            jets = int(jets_)
            mtags = int(mtags_)
            data_filtered = data.loc[(data["nJets"] == jets) & (data["nMtags"] == mtags) & (data["target"] == 0)]

        plt.figure(figsize=(10,8))

        if weightset is None or len(weightset) != 2:
            plt.hist(model.predict(data_filtered[variables_used]), range=(0., 1.), bins=20, histtype="step", density=True)
        else:
            plt.hist(model.predict(data_filtered[variables_used]), range=(0., 1.), bins=20, weights=data_filtered[weightset[0]], histtype="step", label=weightset[0], density=True)
            plt.hist(model.predict(data_filtered[variables_used]), range=(0., 1.), bins=20, histtype="step", label="central", density=True)
            plt.hist(model.predict(data_filtered[variables_used]), range=(0., 1.), bins=20, weights=data_filtered[weightset[1]], histtype="step", label=weightset[1], density=True)
        
        plt.title("Keras output distribution for background events "+ str(jets_) + 'J' + str(mtags_) + 'M' + custom_suffix)
        plt.xlabel("Discriminator output")
        plt.ylabel("Normalised frequency")
        if weightset != None and len(weightset) == 2:
            plt.legend()
        plt.show()

def plot_ROC(label, y_predict, y_test, weight=None, points=50):
    graph = ROC_graph(y_predict, y_test, weight, points)
    x_points = list(graph.GetX())
    y_points = list(graph.GetY())
    x_points.insert(0, 1.)
    y_points.insert(0, 1.)
    plt.plot(x_points, y_points, label="{}: {:.4f}".format(label, graph.Integral(0, -1)+0.5))

def plot_ROC_channels(model, variables_used, data, channels, weightset = None, baseweight = None, custom_suffix = "", raw_variables = ["BDT"]):
    for jets_ , mtags_ in channels:
        if '+' in str(jets_) and '+' in str(mtags_):
            jets = int(jets_[:-1])
            mtags = int(mtags_[:-1])
            data_filtered = data.loc[(data["nJets"] >= jets) & (data["nMtags"] >= mtags)]
        elif '+' in str(mtags_):
            jets = int(jets_)
            mtags = int(mtags_[:-1])
            data_filtered = data.loc[(data["nJets"] == jets) & (data["nMtags"] >= mtags)]
        elif '+' in str(jets_):
            jets = int(jets_[:-1])
            mtags = int(mtags_)
            data_filtered = data.loc[(data["nJets"] >= jets) & (data["nMtags"] == mtags)]
        else:
            jets = int(jets_)
            mtags = int(mtags_)
            data_filtered = data.loc[(data["nJets"] == jets) & (data["nMtags"] == mtags)]
        
        if type(model) == keras.engine.training.Model:
            y_pred = list(model.predict(data_filtered[variables_used]).ravel())
        elif type(model) == keras.wrappers.scikit_learn.KerasClassifier:
            y_pred = list(model.predict_proba(data_filtered[variables_used])[:, 1])
        upper = max(y_pred)
        #print(upper)
        lower = min(y_pred)
        #print(lower)
        y_pred = list((y_pred-lower)/(upper-lower))
        #print(y_pred[0])
        y_test = list(data_filtered["target"].ravel())
        y_pred_rawvar = {}
        for raw_var in raw_variables:
            cache = np.asarray(data_filtered[raw_var].ravel())
            upper = max(cache)
            lower = min(cache)
            cache = list((cache - lower)/(upper-lower))
            y_pred_rawvar[raw_var] = cache
        #print(y_pred, y_test)
        #canvas = pyr.TCanvas("canvas" + str(jets_) + 'J' + str(mtags_) + 'M', "canvas", 800, 600)
        #canvas.cd()
        plt.figure(figsize=(10, 8))
        
        if baseweight is None:
            baseweight = np.asarray([1]*len(y_pred))

        if weightset != None and len(weightset) >= 2:
            if len(weightset) == 3:
                weight_central = list(data_filtered[weightset[2]].ravel())
            else:
                weight_central = [1]*len(y_pred)
            weight_up = list(data_filtered[weightset[0]].ravel())
            weight_down = list(data_filtered[weightset[1]].ravel())
            graph = ROC_graph(y_pred, y_test, weight=weight_central)
            graph_up = ROC_graph(y_pred, y_test, weight=weight_up)
            graph_down = ROC_graph(y_pred, y_test, weight=weight_down)
            graph_rawvar = {}
            for raw_var in y_pred_rawvar.keys():
                graph_cache = ROC_graph(y_pred_rawvar[raw_var], y_test, weight=weight_central)
                graph_cache_up = ROC_graph(y_pred_rawvar[raw_var], y_test, weight=weight_up)
                graph_cache_down = ROC_graph(y_pred_rawvar[raw_var], y_test, weight=weight_down)
                graph_rawvar[raw_var] = (graph_cache, graph_cache_up, graph_cache_down)

            x_points_central = list(graph.GetX())
            x_points_up = list(graph_up.GetX())
            x_points_down = list(graph_down.GetX())
            x_points_rawvar = {}
            for raw_var in y_pred_rawvar.keys():
                x_points_cache = list(graph_rawvar[raw_var][0].GetX())
                x_points_cache_up = list(graph_rawvar[raw_var][1].GetX())
                x_points_cache_down = list(graph_rawvar[raw_var][2].GetX())
                x_points_rawvar[raw_var] = (x_points_cache, x_points_cache_up, x_points_cache_down)

            y_points_central = list(graph.GetY())
            y_points_up = list(graph_up.GetY())
            y_points_down = list(graph_down.GetY())
            y_points_rawvar = {}
            for raw_var in y_pred_rawvar.keys():
                y_points_cache = list(graph_rawvar[raw_var][0].GetY())
                y_points_cache_up = list(graph_rawvar[raw_var][1].GetY())
                y_points_cache_down = list(graph_rawvar[raw_var][2].GetY())
                y_points_rawvar[raw_var] = (y_points_cache, y_points_cache_up, y_points_cache_down)

            x_points_central.insert(0, 1.)
            x_points_up.insert(0, 1.)
            x_points_down.insert(0, 1.)

            y_points_central.insert(0, 1.)
            y_points_up.insert(0, 1.)
            y_points_down.insert(0, 1.)

            for points_rawvar in [x_points_rawvar, y_points_rawvar]:
                for raw_var in points_rawvar:
                    for points in points_rawvar[raw_var]:
                        points.insert(0, 1.)

            plt.plot(x_points_central, y_points_central, label="This model ({:.4f})".format(graph.Integral(0, -1)+0.5))
            plt.plot(x_points_up, y_points_up, label="This model, "+weightset[0]+" ({:.4f})".format(graph_up.Integral(0, -1)+0.5))
            plt.plot(x_points_down, y_points_down, label="This model, "+weightset[1]+" ({:.4f})".format(graph_down.Integral(0, -1)+0.5))

            for raw_var in y_pred_rawvar.keys():
                plt.plot(x_points_rawvar[raw_var][0], y_points_rawvar[raw_var][0], label = "{} ({:.4f})".format(raw_var, graph_rawvar[raw_var][0].Integral(0, -1)+0.5))
                plt.plot(x_points_rawvar[raw_var][1], y_points_rawvar[raw_var][1], label = "{}, {} ({:.4f})".format(raw_var, weightset[0], graph_rawvar[raw_var][0].Integral(0, -1)+0.5))
                plt.plot(x_points_rawvar[raw_var][2], y_points_rawvar[raw_var][2], label = "{}, {} ({:.4f})".format(raw_var, weightset[1], graph_rawvar[raw_var][0].Integral(0, -1)+0.5))

            plt.legend()
            plt.title("ROC curve for "+ str(jets_) + 'J' + str(mtags_) + 'M' + " channel" + custom_suffix)
            plt.xlabel("Background acceptance")
            plt.ylabel("Signal acceptance")
            plt.show()
            #multigraph = pyr.TMultiGraph()
            #multigraph.Add(graph)
            #multigraph.Add(graph_up)
            #multigraph.Add(graph_down)
            #legend = pyr.TLegend(0.7, 0.1, 0.9, 0.3)
            #legend.AddEntry(graph_up, weightset[0])
            #legend.AddEntry(graph, "central")
            #legend.AddEntry(graph_down, weightset[1])
            #multigraph.Draw("plt")
            #multigraph.SetTitle("ROC curve for "+ str(jets_) + 'J' + str(mtags_) + 'M' + " channel")
            #legend.Draw()
            #canvas.Draw()
        else:
            graph = ROC_graph(y_pred, y_test, weight=baseweight)
            graph_rawvar = {}
            for raw_var in y_pred_rawvar.keys():
                graph_rawvar[raw_var] = ROC_graph(y_pred_rawvar[raw_var], y_test, weight=baseweight)

            x_points = list(graph.GetX())
            y_points = list(graph.GetY())
            x_points_rawvar = {}
            y_points_rawvar = {}
            for raw_var in graph_rawvar.keys():
                x_points_rawvar[raw_var] = list(graph_rawvar[raw_var].GetX())
                y_points_rawvar[raw_var] = list(graph_rawvar[raw_var].GetY())

            x_points.insert(0, 1.)
            y_points.insert(0, 1.)
            for raw_var in graph_rawvar.keys():
                x_points_rawvar[raw_var].insert(0, 1.)
                y_points_rawvar[raw_var].insert(0, 1.)

            plt.plot(x_points, y_points, label="This model ({:.4f})".format(graph.Integral(0, -1)+0.5))
            for raw_var in graph_rawvar.keys():
                #print(x_points_rawvar[raw_var])
                #print(y_points_rawvar[raw_var])
                plt.plot(x_points_rawvar[raw_var], y_points_rawvar[raw_var], label="{} ({:.4f})".format(raw_var, graph_rawvar[raw_var].Integral(0, -1)+0.5))
            plt.legend()
            plt.title("ROC curve for "+ str(jets_) + 'J' + str(mtags_) + 'M' + " channel" + custom_suffix)
            plt.xlabel("Background acceptance")
            plt.ylabel("Signal acceptance")
            plt.show()
            #print(list(graph.GetX()))
            #graph.Draw()
            #graph.SetTitle("ROC curve for "+ str(jets_) + 'J' + str(mtags_) + 'M' + " channel")
            #canvas.Draw()

def signal_acc(var_weight_bg, var_weight_sig, background_acc):
    cutoff = 0.
    cache = 0.
    sum_weight_bg = sum([weight for val, weight in var_weight_bg])
    sum_weight_sig = sum([weight for val, weight in var_weight_sig])
    for i in range(len(var_weight_bg)):
        cache += var_weight_bg[i][1]
        if (1-background_acc) < cache/sum_weight_bg:
            cutoff = var_weight_bg[i][0]
            break
    
    cache = 0.
    for i in range(len(var_weight_sig)):
        cache += var_weight_sig[i][1]
        if var_weight_sig[i][0] > cutoff:
            break

    return 1. - cache/sum_weight_sig

def signal_acceptance_report(model, variables_used, data, channels, background_acc_list, weightsets, raw_variables = ["BDT"], baseweight_column = None):
    for jets_ , mtags_ in channels:
        if '+' in str(jets_) and '+' in str(mtags_):
            jets = int(jets_[:-1])
            mtags = int(mtags_[:-1])
            data_filtered = data.loc[(data["nJets"] >= jets) & (data["nMtags"] >= mtags)]
        elif '+' in str(mtags_):
            jets = int(jets_)
            mtags = int(mtags_[:-1])
            data_filtered = data.loc[(data["nJets"] == jets) & (data["nMtags"] >= mtags)]
        elif '+' in str(jets_):
            jets = int(jets_[:-1])
            mtags = int(mtags_)
            data_filtered = data.loc[(data["nJets"] >= jets) & (data["nMtags"] == mtags)]
        else:
            jets = int(jets_)
            mtags = int(mtags_)
            data_filtered = data.loc[(data["nJets"] == jets) & (data["nMtags"] == mtags)]
        
        if type(model) == keras.engine.training.Model:
            y_pred_bg  = list(model.predict(data_filtered.loc[data_filtered["target"]==0][variables_used]).ravel())
            y_pred_sig = list(model.predict(data_filtered.loc[data_filtered["target"]==1][variables_used]).ravel())
        elif type(model) == keras.wrappers.scikit_learn.KerasClassifier:
            print("Keras sklearn classifier used")
            y_pred_bg  = list(model.predict_proba(data_filtered.loc[data_filtered["target"]==0][variables_used])[:, 1])
            y_pred_sig = list(model.predict_proba(data_filtered.loc[data_filtered["target"]==1][variables_used])[:, 1])
        
        y_raw_bg = {}
        y_raw_sig = {}
        for var in raw_variables:
            y_raw_bg[var] = list(data_filtered.loc[data_filtered["target"]==0][var].ravel())
            y_raw_sig[var] = list(data_filtered.loc[data_filtered["target"]==1][var].ravel())
        
        print("For {}J{}M channel".format(jets_, mtags_))

        for weightset in weightsets:
            print("\tThis", end='')
            for var in raw_variables:
                print("\t\t\t{}".format(var), end='')
            print()

            if baseweight_column == None:
                weight_up_bg = list(data_filtered.loc[data_filtered["target"]==0][weightset[0]].ravel())
                weight_up_sig = list(data_filtered.loc[data_filtered["target"]==1][weightset[0]].ravel())
                weight_down_bg = list(data_filtered.loc[data_filtered["target"]==0][weightset[1]].ravel())
                weight_down_sig = list(data_filtered.loc[data_filtered["target"]==1][weightset[1]].ravel())
                if len(weightset) == 3:
                    weight_central_bg = list(data_filtered.loc[data_filtered["target"]==0][weightset[2]].ravel())
                    weight_central_sig = list(data_filtered.loc[data_filtered["target"]==1][weightset[2]].ravel())
                    for i in range(len(raw_variables)+1):
                        print("\t{}\t{}\t{}".format(weightset[2], weightset[0], weightset[1]), end='')
                else:
                    weight_central_bg = [1]*len(y_pred_bg)
                    weight_central_sig = [1]*len(y_pred_sig)
                    for i in range(len(raw_variables)+1):
                        print("\tCentral\t{}\t{}".format(weightset[0], weightset[1]), end='')
                print()
            else:
                weight_up_bg = list(np.asarray(data_filtered.loc[data_filtered["target"]==0][weightset[0]].ravel())*np.asarray(data_filtered.loc[data_filtered["target"]==0][baseweight_column].ravel()))
                weight_up_sig = list(np.asarray(data_filtered.loc[data_filtered["target"]==1][weightset[0]].ravel())*np.asarray(data_filtered.loc[data_filtered["target"]==1][baseweight_column].ravel()))
                weight_down_bg = list(np.asarray(data_filtered.loc[data_filtered["target"]==0][weightset[1]].ravel())*np.asarray(data_filtered.loc[data_filtered["target"]==0][baseweight_column].ravel()))
                weight_down_sig = list(np.asarray(data_filtered.loc[data_filtered["target"]==1][weightset[1]].ravel())*np.asarray(data_filtered.loc[data_filtered["target"]==1][baseweight_column].ravel()))
                if len(weightset) == 3:
                    weight_central_bg = list(np.asarray(data_filtered.loc[data_filtered["target"]==0][weightset[2]].ravel())*np.asarray(data_filtered.loc[data_filtered["target"]==0][baseweight_column].ravel()))
                    weight_central_sig = list(np.asarray(data_filtered.loc[data_filtered["target"]==1][weightset[2]].ravel())*np.asarray(data_filtered.loc[data_filtered["target"]==1][baseweight_column].ravel()))
                    for i in range(len(raw_variables)+1):
                        print("\t{}\t{}\t{}".format(weightset[2], weightset[0], weightset[1]), end='')
                else:
                    weight_central_bg = list(data_filtered.loc[data_filtered["target"]==0][baseweight_column].ravel())
                    weight_central_sig = list(data_filtered.loc[data_filtered["target"]==1][baseweight_column].ravel())
                    for i in range(len(raw_variables)+1):
                        print("\tCentral\t{}\t{}".format(weightset[0], weightset[1]), end='')
                print()
            
            sum_weight_up_bg = sum(weight_up_bg)
            sum_weight_up_sig = sum(weight_up_sig)
            sum_weight_down_bg = sum(weight_down_bg)
            sum_weight_down_sig = sum(weight_down_sig)
            sum_weight_central_bg = sum(weight_central_bg)
            sum_weight_central_sig = sum(weight_central_sig)

            y_pred_up_bg = list(zip(y_pred_bg, weight_up_bg))
            y_pred_up_sig = list(zip(y_pred_sig, weight_up_sig))
            y_pred_down_bg = list(zip(y_pred_bg, weight_down_bg))
            y_pred_down_sig = list(zip(y_pred_sig, weight_down_sig))
            y_pred_central_bg = list(zip(y_pred_bg, weight_central_bg))
            y_pred_central_sig = list(zip(y_pred_sig, weight_central_sig))

            y_raw_up_bg = {}
            y_raw_down_bg = {}
            y_raw_central_bg = {}
            y_raw_up_sig = {}
            y_raw_down_sig = {}
            y_raw_central_sig = {}

            for var in raw_variables:
                y_raw_up_bg[var] = list(zip(y_raw_bg[var], weight_up_bg))
                y_raw_down_bg[var] = list(zip(y_raw_bg[var], weight_down_bg))
                y_raw_central_bg[var] = list(zip(y_raw_bg[var], weight_central_bg))
                y_raw_up_sig[var] = list(zip(y_raw_sig[var], weight_up_sig))
                y_raw_down_sig[var] = list(zip(y_raw_sig[var], weight_down_sig))
                y_raw_central_sig[var] = list(zip(y_raw_sig[var], weight_central_sig))

            for l in [y_pred_up_bg, y_pred_up_sig, y_pred_down_bg, y_pred_down_sig, y_pred_central_bg, y_pred_central_sig]:
                l.sort(key = lambda x: x[0])
            for l in [y_raw_up_bg, y_raw_down_bg, y_raw_central_bg, y_raw_up_sig, y_raw_down_sig, y_raw_central_sig]:
                for var in l.keys():
                    l[var].sort(key = lambda x: x[0])

            for background_acc in background_acc_list:
                raw_signal_acc = {}
                raw_signal_acc_up = {}
                raw_signal_acc_down = {}

                pred_signal_acc = signal_acc(y_pred_central_bg, y_pred_central_sig, background_acc)
                pred_signal_acc_up = signal_acc(y_pred_up_bg, y_pred_up_sig, background_acc)
                pred_signal_acc_down = signal_acc(y_pred_down_bg, y_pred_down_sig, background_acc)

                for var in raw_variables:
                    raw_signal_acc[var] = signal_acc(y_raw_central_bg[var], y_raw_central_sig[var], background_acc)
                    raw_signal_acc_up[var] = signal_acc(y_raw_up_bg[var], y_raw_up_sig[var], background_acc)
                    raw_signal_acc_down[var] = signal_acc(y_raw_down_bg[var], y_raw_down_sig[var], background_acc)

                print("{:.2f}\t{:.4f}\t{:.4f}\t{:.4f}".format(background_acc, pred_signal_acc, pred_signal_acc_up, pred_signal_acc_down), end='')
                for var in raw_variables:
                    print("\t{:.4f}\t{:.4f}\t{:.4f}".format(raw_signal_acc[var], raw_signal_acc_up[var], raw_signal_acc_down[var]), end='')
                print()

def signal_acceptance(model, variables_used, data, channels, background_acc, weightset = None):
    for jets_ , mtags_ in channels:
        if '+' in str(jets_) and '+' in str(mtags_):
            jets = int(jets_[:-1])
            mtags = int(mtags_[:-1])
            data_filtered = data.loc[(data["nJets"] >= jets) & (data["nMtags"] >= mtags)]
        elif '+' in str(mtags_):
            jets = int(jets_)
            mtags = int(mtags_[:-1])
            data_filtered = data.loc[(data["nJets"] == jets) & (data["nMtags"] >= mtags)]
        elif '+' in str(jets_):
            jets = int(jets_[:-1])
            mtags = int(mtags_)
            data_filtered = data.loc[(data["nJets"] >= jets) & (data["nMtags"] == mtags)]
        else:
            jets = int(jets_)
            mtags = int(mtags_)
            data_filtered = data.loc[(data["nJets"] == jets) & (data["nMtags"] == mtags)]
        
        y_pred_bg  = list(model.predict(data_filtered.loc[data_filtered["target"]==0][variables_used]).ravel())
        y_pred_sig = list(model.predict(data_filtered.loc[data_filtered["target"]==1][variables_used]).ravel())
        y_pred_bdt_bg = list(data_filtered.loc[data_filtered["target"]==0]["BDT"].ravel())
        y_pred_bdt_sig = list(data_filtered.loc[data_filtered["target"]==1]["BDT"].ravel())
        
        print("For {}J{}M channel".format(jets_, mtags_))
    
        if weightset != None and len(weightset) >= 2:
            weight_up_bg = list(data_filtered.loc[data_filtered["target"]==0][weightset[0]].ravel())
            weight_up_sig = list(data_filtered.loc[data_filtered["target"]==1][weightset[0]].ravel())
            weight_down_bg = list(data_filtered.loc[data_filtered["target"]==0][weightset[1]].ravel())
            weight_down_sig = list(data_filtered.loc[data_filtered["target"]==1][weightset[1]].ravel())
            if len(weightset) == 3:
                weight_central_bg = list(data_filtered.loc[data_filtered["target"]==0][weightset[2]].ravel())
                weight_central_sig = list(data_filtered.loc[data_filtered["target"]==1][weightset[2]].ravel())
            else:
                weight_central_bg = [1]*len(y_pred_bg)
                weight_central_sig = [1]*len(y_pred_sig)
            
            sum_weight_up_bg = sum(weight_up_bg)
            sum_weight_up_sig = sum(weight_up_sig)
            sum_weight_down_bg = sum(weight_down_bg)
            sum_weight_down_sig = sum(weight_down_sig)
            sum_weight_central_bg = sum(weight_central_bg)
            sum_weight_central_sig = sum(weight_central_sig)

            y_pred_up_bg = list(zip(y_pred_bg, weight_up_bg))
            y_pred_up_sig = list(zip(y_pred_sig, weight_up_sig))
            y_pred_down_bg = list(zip(y_pred_bg, weight_down_bg))
            y_pred_down_sig = list(zip(y_pred_sig, weight_down_sig))
            y_pred_central_bg = list(zip(y_pred_bg, weight_central_bg))
            y_pred_central_sig = list(zip(y_pred_sig, weight_central_sig))

            y_bdt_up_bg = list(zip(y_pred_bdt_bg, weight_up_bg))
            y_bdt_up_sig = list(zip(y_pred_bdt_sig, weight_up_sig))
            y_bdt_down_bg = list(zip(y_pred_bdt_bg, weight_down_bg))
            y_bdt_down_sig = list(zip(y_pred_bdt_sig, weight_down_sig))
            y_bdt_central_bg = list(zip(y_pred_bdt_bg, weight_central_bg))
            y_bdt_central_sig = list(zip(y_pred_bdt_sig, weight_central_sig))

            for l in [y_pred_up_bg, y_pred_up_sig, y_pred_down_bg, y_pred_down_sig, y_pred_central_bg, y_pred_central_sig, y_bdt_up_bg, y_bdt_up_sig, y_bdt_down_bg, y_bdt_down_sig, y_bdt_central_bg, y_bdt_central_sig]:
                l.sort(key = lambda x: x[0])

            cutoff = y_pred_central_bg[math.floor(len(y_pred_central_bg)*(1-background_acc))][0]
            cache = 0
            for i in range(len(y_pred_central_sig)):
                if y_pred_central_sig[i][0] > cutoff:
                    cache = i
                    break
            
            signal_acc = 1. - cache/sum_weight_central_sig

            cutoff = y_bdt_central_bg[math.floor(len(y_bdt_central_bg)*(1-background_acc))][0]
            cache = 0
            for i in range(len(y_bdt_central_sig)):
                if y_bdt_central_sig[i][0] > cutoff:
                    cache = i
                    break
            
            bdt_signal_acc = 1. - cache/len(y_pred_bdt_sig)

            cutoff = 0.
            cache = 0.
            for i in range(len(y_pred_up_bg)):
                cache += y_pred_up_bg[i][1]
                if (1-background_acc) < cache/sum_weight_up_bg:
                    cutoff = y_pred_up_bg[i][0]
                    break
            
            cache = 0.
            for i in range(len(y_pred_up_sig)):
                cache += y_pred_up_sig[i][1]
                if y_pred_up_sig[i][0] > cutoff:
                    break

            signal_acc_up = 1. - cache/sum_weight_up_sig

            cutoff = 0.
            cache = 0.
            for i in range(len(y_bdt_up_bg)):
                cache += y_bdt_up_bg[i][1]
                if (1-background_acc) < cache/sum_weight_up_bg:
                    cutoff = y_bdt_up_bg[i][0]
                    break
            
            cache = 0.
            for i in range(len(y_bdt_up_sig)):
                cache += y_bdt_up_sig[i][1]
                if y_bdt_up_sig[i][0] > cutoff:
                    break

            bdt_signal_acc_up = 1. - cache/sum_weight_up_sig
            
            cutoff = 0.
            cache = 0.
            for i in range(len(y_pred_down_bg)):
                cache += y_pred_down_bg[i][1]
                if (1-background_acc) < cache/sum_weight_down_bg:
                    cutoff = y_pred_down_bg[i][0]
                    break
            
            cache = 0.
            for i in range(len(y_pred_down_sig)):
                cache += y_pred_down_sig[i][1]
                if y_pred_down_sig[i][0] > cutoff:
                    break

            signal_acc_down = 1. - cache/sum_weight_down_sig

            cutoff = 0.
            cache = 0.
            for i in range(len(y_bdt_down_bg)):
                cache += y_bdt_down_bg[i][1]
                if (1-background_acc) < cache/sum_weight_down_bg:
                    cutoff = y_bdt_down_bg[i][0]
                    break
            
            cache = 0.
            for i in range(len(y_bdt_down_sig)):
                cache += y_bdt_down_sig[i][1]
                if y_bdt_down_sig[i][0] > cutoff:
                    break

            bdt_signal_acc_down = 1. - cache/sum_weight_down_sig

            print("Central, {}, {}: {:.4f} + {:.4f} - {:.4f}".format(weightset[0], weightset[1], signal_acc, max(signal_acc_up, signal_acc_down) - signal_acc, signal_acc - min(signal_acc_up, signal_acc_down)))
            print("{}: {:.4f}".format(weightset[0], signal_acc_up))
            print("{}: {:.4f}".format(weightset[1], signal_acc_down))
            print("Central, {}, {}: {:.4f} + {:.4f} - {:.4f}".format(weightset[0], weightset[1], bdt_signal_acc, max(bdt_signal_acc_up, bdt_signal_acc_down) - bdt_signal_acc, bdt_signal_acc - min(bdt_signal_acc_up, bdt_signal_acc_down)))
            print("Official MVA, {}: {:.4f}".format(weightset[0], bdt_signal_acc_up))
            print("Official MVA, {}: {:.4f}".format(weightset[1], bdt_signal_acc_down))

        else:
            cutoff = y_pred_bg[math.floor(len(y_pred_bg)*(1-background_acc))]
            cache = 0
            for i in range(len(y_pred_sig)):
                if y_pred_sig[i] > cutoff:
                    cache = i
                    break
            
            signal_acc = 1. - cache/len(y_pred_sig)

            cutoff = y_pred_bdt_bg[math.floor(len(y_pred_bdt_bg)*(1-background_acc))]
            cache = 0
            for i in range(len(y_pred_bdt_sig)):
                if y_pred_bdt_sig[i] > cutoff:
                    cache = i
                    break
            
            bdt_signal_acc = 1. - cache/len(y_pred_bdt_sig)

            print("Central: {:.4f}".format(signal_acc))
            print("Official MVA, central: {:.4f}".format(bdt_signal_acc))

def plot_sig_bg_dist(data_sig, data_bg, var, range_, density=True, bins = 20):
    plt.hist(data_bg[var], range=range_, bins=bins, density=density, histtype="step", label="background")
    plt.hist(data_sig[var], range=range_, bins=bins, density=density, histtype="step", label="signal")
    plt.xlabel(var)
    plt.legend()

def overlap_coeff_dataframe(data_sig, data_bg, var, bins=1000):
    arr_sig = np.asarray(data_sig[var])
    arr_bg = np.asarray(data_bg[var])
    return overlap_coeff(arr_sig, arr_bg, bins)

def overlap_coeff(arr_sig, arr_bg, bins=1000, weight_sig = None, weight_bg = None):
    if weight_sig is None:
        weight_sig_sum = len(arr_sig)
    else:
        weight_sig_sum = sum(weight_sig)
    if weight_bg is None:
        weight_bg_sum = len(arr_bg)
    else:
        weight_bg_sum = sum(weight_bg)
    range_min = min(arr_sig.min(), arr_bg.min())
    range_max = min(arr_sig.max(), arr_bg.max())
    bin_width = (range_max - range_min)/bins
    lower_bound = range_min
    min_arr = np.empty(bins)
    for b in range(bins):
        higher_bound = lower_bound + bin_width
        if weight_sig is not None:
            freq_sig = np.ma.compressed(np.ma.masked_where((arr_sig<lower_bound) | (arr_sig>=higher_bound), weight_sig)).sum()/weight_sig_sum
        else:
            freq_sig = np.ma.masked_where((arr_sig<lower_bound) | (arr_sig>=higher_bound), arr_sig).count()/weight_sig_sum
        if weight_bg is not None:
            freq_bg = np.ma.compressed(np.ma.masked_where((arr_bg<lower_bound) | (arr_bg>=higher_bound), weight_bg)).sum()/weight_bg_sum
        else:
            freq_bg = np.ma.masked_where((arr_bg<lower_bound) | (arr_bg>=higher_bound), arr_bg).count()/weight_bg_sum
        min_arr[b] = np.min((freq_sig, freq_bg))
        lower_bound = higher_bound
    return min_arr.sum()