import numpy as np
import ROOT as pyr
import root_numpy
import pickle
import Toolset
import pandas as pd
import itertools
import math

# To install root_numpy
# sudo ROOTSYS=$ROOTSYS pip3 install root_numpy

def add_trijet_comb(dataset):
    trijet_comb = []
    for eventNo in range(dataset.shape[0]):
        #print("Starting event {}".format(eventNo))
        print("Processing trijet combination for event {}".format(eventNo), end='\r')
        delta = 99
        usingcombi = None
        ht_htx_goal = dataset["HT"][eventNo]-dataset["HTX"][eventNo]
        possible_jets = []
        for i in range(len(dataset["jetvec"][eventNo])):
            if dataset["jetvec"][eventNo][i][0] < ht_htx_goal:
                possible_jets.append(i)
        for combi in itertools.combinations(possible_jets, 3):
            if abs(sum([dataset["jetvec"][eventNo][l][0] for l in combi])-ht_htx_goal) < delta:
                delta = abs(sum([dataset["jetvec"][eventNo][l][0] for l in combi])-(dataset["HT"][eventNo]-dataset["HTX"][eventNo]))
                #print(delta)
                usingcombi = combi
        if delta > 0.1:
            print("Warning: delta is {} in event {}".format(delta, eventNo))
        if usingcombi == None:
            print("Error! No combinations found at event {}!".format(eventNo))
            raise ArithmeticError()
        trijet_comb.append(usingcombi)
    print("Trijet combination for {} events recorded in dataframe.".format(dataset.shape[0]))
    dataset["trijet_comb"] = trijet_comb

branchlist_ = [
    "multitopness", "HTH", "HTb", "HTRat", "jetvec", "1stjetpt", "2ndjetpt", ("Max$(jetvec[2][0])", "3rdjetpt"), ("Max$(jetvec[3][0])", "4thjetpt"), "5thjetpt", "6thjetpt", "jet5and6pt", "LeadingBJetPt", "SumJetMassX", "HTX", "angletop1top2", "angletoplep", 
    "LeptonPt", "LeptonEta", "leptonphi", "leptonIso", "csvJetcsv1", "csvJetcsv2", "csvJetcsv3", "csvJetcsv4", 
    "csvJetpt1", "csvJetpt2", "csvJetpt3", "csvJetpt4", "PU", "met", 
    ("Max$(abs(weight))", "MEup"), ("Min$(abs(weight))", "MEdown"), "SFlepton", "toprew", "GenWeight", "SFtrig", "HT", "BDT", "BDT1", "nJets", "nLtags", "nMtags", "nTtags",
    ("Max$(csvrsw[0])", "CSVRScentre"), ("Max$(csvrsw[3])", "CSVRSup"), ("Max$(csvrsw[4])", "CSVRSdown"), ("ttxrew", "TTXcentre"), ("Max$(abs(ttxw))", "TTXup"), ("Min$(abs(ttxw))", "TTXdown"), ("Max$(abs(pdfw))", "PDFup"), ("Min$(abs(pdfw))", "PDFdown"), "SFPU", "SFPU_up", "SFPU_down", "Eventnr"]

branchlist = []
branchlist_name = []

for entity in branchlist_:
    if type(entity) is tuple:
        if len(entity) == 2:
            branchlist.append(entity[0])
            branchlist_name.append(entity[1])
        else:
            print("Error processing entity {}. Exiting...".format(entity))
            exit()
    else:
        branchlist.append(entity)
        branchlist_name.append(entity)

selection_el = "((HLT_Ele32_eta2p1_WPTight_Gsf==1) && met > 50 && HT > 500 && fabs(LeptonEta)<2.1 && LeptonPt > 35) && ((nJets == 8 && nMtags >= 3) || nJets >= 9) && (csvJetpt3 >= 0. && csvJetpt4 >= 0.)"
selection_el_loose = "((HLT_Ele32_eta2p1_WPTight_Gsf==1) && met > 50 && HT > 500 && fabs(LeptonEta)<2.1 && LeptonPt > 35) && (nJets >= 8) && (csvJetpt3 >= 0. && csvJetpt4 >= 0.)"
selection_mu = "((HLT_IsoMu24==1||HLT_IsoTkMu24==1) && met > 50 && HT > 450 && fabs(LeptonEta)<2.1 && leptonIso<0.15 && LeptonPt > 26) && ((nJets == 8 && nMtags >= 3) || nJets >= 9) && (csvJetpt3 >= 0. && csvJetpt4 >= 0.)"
selection_mu_loose = "((HLT_IsoMu24==1||HLT_IsoTkMu24==1) && met > 50 && HT > 450 && fabs(LeptonEta)<2.1 && leptonIso<0.15 && LeptonPt > 26) && (nJets >= 7) && (csvJetpt3 >= 0. && csvJetpt4 >= 0.)"

sig_el_craneen         = "/mnt/e/new_Craneen__El/Craneen_ttttNLO_Run2_TopTree_Study.root"
sig_el_craneen_jesup   = "/mnt/e/new_Craneen__El/Craneen_ttttNLO_totaljesup_Run2_TopTree_Study.root"
sig_el_craneen_jesdown = "/mnt/e/new_Craneen__El/Craneen_ttttNLO_totaljesdown_Run2_TopTree_Study.root"
sig_el_craneen_jerup   = "/mnt/e/new_Craneen__El/Craneen_ttttNLO_jerup_Run2_TopTree_Study.root"
sig_el_craneen_jerdown = "/mnt/e/new_Craneen__El/Craneen_ttttNLO_jerdown_Run2_TopTree_Study.root"

bg_el_craneen          = "/mnt/e/new_Craneen__El/Craneen_TTJetsFilt_powheg_central_Run2_TopTree_Study.root"
#bg_el_craneen         = "/mnt/e/new_Craneen__El/Craneen_TTJetsFilt_powheg_Run2_TopTree_Study.root"
#bg_el_craneen         = "/mnt/e/new_Craneen__El/Craneen_TTJetsFilt_powheg_mixture_Run2_TopTree_Study.root"
bg_el_craneen_jesup    = "/mnt/e/new_Craneen__El/Craneen_TTJetsFilt_powheg_totaljesup_Run2_TopTree_Study.root"
bg_el_craneen_jesdown  = "/mnt/e/new_Craneen__El/Craneen_TTJetsFilt_powheg_totaljesdown_Run2_TopTree_Study.root"
bg_el_craneen_jerup    = "/mnt/e/new_Craneen__El/Craneen_TTJetsFilt_powheg_jerup_Run2_TopTree_Study.root"
bg_el_craneen_jerdown  = "/mnt/e/new_Craneen__El/Craneen_TTJetsFilt_powheg_jerdown_Run2_TopTree_Study.root"

sig_mu_craneen         = "/mnt/e/new_Craneen__Mu/Craneen_ttttNLO_Run2_TopTree_Study.root"
sig_mu_craneen_jesup   = "/mnt/e/new_Craneen__Mu/Craneen_ttttNLO_totaljesup_Run2_TopTree_Study.root"
sig_mu_craneen_jesdown = "/mnt/e/new_Craneen__Mu/Craneen_ttttNLO_totaljesdown_Run2_TopTree_Study.root"
sig_mu_craneen_jerup   = "/mnt/e/new_Craneen__Mu/Craneen_ttttNLO_jerup_Run2_TopTree_Study.root"
sig_mu_craneen_jerdown = "/mnt/e/new_Craneen__Mu/Craneen_ttttNLO_jerdown_Run2_TopTree_Study.root"

bg_mu_craneen          = "/mnt/e/new_Craneen__Mu/Craneen_TTJetsFilt_powheg_central_Run2_TopTree_Study.root"
#bg_mu_craneen         = "/mnt/e/new_Craneen__Mu/Craneen_TTJetsFilt_powheg_Run2_TopTree_Study.root"
#bg_mu_craneen         = "/mnt/e/new_Craneen__Mu/Craneen_TTJetsFilt_powheg_mixture_Run2_TopTree_Study.root"
bg_mu_craneen_jesup    = "/mnt/e/new_Craneen__Mu/Craneen_TTJetsFilt_powheg_totaljesup_Run2_TopTree_Study.root"
bg_mu_craneen_jesdown  = "/mnt/e/new_Craneen__Mu/Craneen_TTJetsFilt_powheg_totaljesdown_Run2_TopTree_Study.root"
bg_mu_craneen_jerup    = "/mnt/e/new_Craneen__Mu/Craneen_TTJetsFilt_powheg_jerup_Run2_TopTree_Study.root"
bg_mu_craneen_jerdown  = "/mnt/e/new_Craneen__Mu/Craneen_TTJetsFilt_powheg_jerdown_Run2_TopTree_Study.root"

# print(branchlist)

# df_sig_el = Toolset.to_pandas(sig_el_craneen, "Craneen__El", selection_el, branchlist, branchlist_name, isSignal=True, add_weight=True)
# df_bg_el  = Toolset.to_pandas(bg_el_craneen, "Craneen__El", selection_el, branchlist, branchlist_name, isSignal=False, add_weight=True)
# df_sig_mu = Toolset.to_pandas(sig_mu_craneen, "Craneen__Mu", selection_mu, branchlist, branchlist_name, isSignal=True, add_weight=True)
# df_bg_mu  = Toolset.to_pandas(bg_mu_craneen, "Craneen__Mu", selection_mu, branchlist, branchlist_name, isSignal=False, add_weight=True)
# 
# df_sig_el["fromMu"] = 0
# df_bg_el["fromMu"] = 0
# df_sig_mu["fromMu"] = 1
# df_bg_mu["fromMu"] = 1
# 
# dataframe = pd.concat([df_sig_el, df_bg_el, df_sig_mu, df_bg_mu], ignore_index=True)
# dataframe["jes"] = 0
# 
# dataframe.to_pickle("fourtops_8J3M_pd.p")

delimiter = "===================================="
print(delimiter)
print("Starting central samples")

df_sig_el_loose = Toolset.to_pandas(sig_el_craneen, "Craneen__El", selection_el_loose, branchlist, branchlist_name, isSignal=True, add_weight=True)
df_bg_el_loose  = Toolset.to_pandas(bg_el_craneen, "Craneen__El", selection_el_loose, branchlist, branchlist_name, isSignal=False, add_weight=True)
df_sig_mu_loose = Toolset.to_pandas(sig_mu_craneen, "Craneen__Mu", selection_mu_loose, branchlist, branchlist_name, isSignal=True, add_weight=True)
df_bg_mu_loose  = Toolset.to_pandas(bg_mu_craneen, "Craneen__Mu", selection_mu_loose, branchlist, branchlist_name, isSignal=False, add_weight=True)

for trijet in ["trijet1stpass", "trijet2ndpass", "trijet3rdpass"]:
    df_sig_el_loose[trijet] = [np.asarray(row[0]) for row in root_numpy.root2array(sig_el_craneen, "Craneen__El", selection=selection_el_loose, branches=[trijet], step=1)]
    df_bg_el_loose[trijet]  = [np.asarray(row[0]) for row in root_numpy.root2array(bg_el_craneen,  "Craneen__El", selection=selection_el_loose, branches=[trijet], step=1)]
    df_sig_mu_loose[trijet] = [np.asarray(row[0]) for row in root_numpy.root2array(sig_mu_craneen, "Craneen__Mu", selection=selection_mu_loose, branches=[trijet], step=1)]
    df_bg_mu_loose[trijet]  = [np.asarray(row[0]) for row in root_numpy.root2array(bg_mu_craneen,  "Craneen__Mu", selection=selection_mu_loose, branches=[trijet], step=1)]

df_sig_el_loose["fromMu"] = 0
df_bg_el_loose["fromMu"] = 0
df_sig_mu_loose["fromMu"] = 1
df_bg_mu_loose["fromMu"] = 1

dataframe_loose = pd.concat([df_sig_el_loose, df_bg_el_loose, df_sig_mu_loose, df_bg_mu_loose], ignore_index=True)
# print("Adding trijet combination. Please wait...")
# print("You know what, do something else. Trust me.")
# add_trijet_comb(dataframe_loose)
dataframe_loose["jes"] = 0

### Some extra variables
def add_variables(dataframe_loose):
    dataframe_loose["H"] = dataframe_loose["HT"]/dataframe_loose["HTH"]

    dataframe_loose["pt3HT"] = dataframe_loose["3rdjetpt"]/dataframe_loose["HT"]
    dataframe_loose["pt4HT"] = dataframe_loose["4thjetpt"]/dataframe_loose["HT"]

    sphericity = []
    for event in dataframe_loose["jetvec"]:
        sphericity_matrix = np.zeros([3, 3])
        sumP2 = 0
        for jet in event:
            sumP2 += (jet[0]*math.cosh(jet[1]))**2
            sphericity_matrix[0][0] += (jet[0]*math.cos(jet[2]))**2
            sphericity_matrix[0][1] += (jet[0]*math.cos(jet[2]))*(jet[0]*math.sin(jet[2]))
            sphericity_matrix[0][2] += (jet[0]*math.cos(jet[2]))*(jet[0]*math.sinh(jet[1]))
            sphericity_matrix[1][0] += (jet[0]*math.cos(jet[2]))*(jet[0]*math.sin(jet[2]))
            sphericity_matrix[1][1] += (jet[0]*math.sin(jet[2]))**2
            sphericity_matrix[1][2] += (jet[0]*math.sin(jet[2]))*(jet[0]*math.sinh(jet[1]))
            sphericity_matrix[2][0] += (jet[0]*math.cos(jet[2]))*(jet[0]*math.sinh(jet[1]))
            sphericity_matrix[2][1] += (jet[0]*math.sin(jet[2]))*(jet[0]*math.sinh(jet[1]))
            sphericity_matrix[2][2] += (jet[0]*math.sinh(jet[1]))**2
        sphericity_matrix = sphericity_matrix/sumP2
        eigenvals, _ = np.linalg.eigh(sphericity_matrix)
        eigenvals.sort()
        sphericity.append(sum(eigenvals[0:2])*3/2)
    dataframe_loose["sphericity"] = sphericity

    def invariant_mass(pt1, eta1, phi1, E1, pt2, eta2, phi2, E2):
        return math.sqrt((E1+E2)**2 - (pt1*math.sinh(eta1) + pt2*math.sinh(eta2))**2 - pt1**2 - pt2**2 - 2*pt1*pt2*math.cos(phi1-phi2))

    invmass_34 = []
    invmass_35 = []
    invmass_36 = []
    invmass_45 = []
    invmass_46 = []
    invmass_56 = []

    for event in dataframe_loose["jetvec"]:
        csvjetvec = sorted(event, key=lambda x: -x[3])
        nonbjet = csvjetvec[2:6]
        massnonb_lambda = lambda n1, n2: \
        invariant_mass(nonbjet[n1][0], nonbjet[n1][1], nonbjet[n1][2], nonbjet[n1][4], \
                    nonbjet[n2][0], nonbjet[n2][1], nonbjet[n2][2], nonbjet[n2][4])
        invmass_34.append(massnonb_lambda(0, 1))
        invmass_35.append(massnonb_lambda(0, 2))
        invmass_36.append(massnonb_lambda(0, 3))
        invmass_45.append(massnonb_lambda(1, 2))
        invmass_46.append(massnonb_lambda(1, 3))
        invmass_56.append(massnonb_lambda(2, 3))

    dataframe_loose["invmass_34"] = invmass_34
    dataframe_loose["invmass_35"] = invmass_35
    dataframe_loose["invmass_36"] = invmass_36
    dataframe_loose["invmass_45"] = invmass_45
    dataframe_loose["invmass_46"] = invmass_46
    dataframe_loose["invmass_56"] = invmass_56

    csv3rdjetpt = []
    csv4thjetpt = []

    for event in dataframe_loose["jetvec"]:
        csvjetvec = sorted(event, key=lambda x: -x[3])
        csv3rdjetpt.append(csvjetvec[2][0])
        csv4thjetpt.append(csvjetvec[3][0])
        
    dataframe_loose["csv3rdjetpt"] = csv3rdjetpt
    dataframe_loose["csv4thjetpt"] = csv4thjetpt

    ht2m = []

    for event in dataframe_loose["jetvec"]:
        ht = 0
        pt_bjet = []
        for jet in event:
            ht += jet[0]
            if jet[3] > 0.8484:
                pt_bjet.append(jet[0])
        pt_bjet.sort()
        ht2m.append(ht-sum(pt_bjet[-2:]))

    dataframe_loose["HT2M"] = ht2m

    mean_csv = []
    for event in dataframe_loose["jetvec"]:
        csv_sum = 0.
        jet_count = 0
        for jet in sorted(event, key=lambda x:-x[3]):
            if jet[3] > 0: 
                csv_sum += jet[3]
                jet_count += 1
        mean_csv.append(csv_sum/len(event))

    dataframe_loose["mean_csv"] = mean_csv

    trijet1st_invmass = []
    for _, event in dataframe_loose.iterrows():
        px = 0
        py = 0
        pz = 0
        E = 0
        for jet in event["trijet1stpass"]:
            px += jet[0]*math.cos(jet[2])
            py += jet[0]*math.sin(jet[2])
            pz += jet[0]*math.sinh(jet[1])
            E += jet[4]
        trijet1st_invmass.append(math.sqrt(E**2 - px**2 - py**2 - pz**2))
    dataframe_loose["trijet1st_invmass"] = trijet1st_invmass

    trijet2nd_invmass = []
    for _, event in dataframe_loose.iterrows():
        px = 0
        py = 0
        pz = 0
        E = 0
        for jet in event["trijet2ndpass"]:
            px += jet[0]*math.cos(jet[2])
            py += jet[0]*math.sin(jet[2])
            pz += jet[0]*math.sinh(jet[1])
            E += jet[4]
        trijet2nd_invmass.append(math.sqrt(E**2 - px**2 - py**2 - pz**2))
    dataframe_loose["trijet2nd_invmass"] = trijet2nd_invmass

    trijet3rd_invmass = []
    for _, event in dataframe_loose.iterrows():
        px = 0
        py = 0
        pz = 0
        E = 0
        for jet in event["trijet3rdpass"]:
            px += jet[0]*math.cos(jet[2])
            py += jet[0]*math.sin(jet[2])
            pz += jet[0]*math.sinh(jet[1])
            E += jet[4]
        if (E**2 - px**2 - py**2 - pz**2) < 0:
            trijet3rd_invmass.append(-10.)
        else:
            trijet3rd_invmass.append(math.sqrt(E**2 - px**2 - py**2 - pz**2))
    dataframe_loose["trijet3rd_invmass"] = trijet3rd_invmass

    angletoplep = []

    def deltaR(eta1, phi1, eta2, phi2):
        return math.sqrt((eta1-eta2)**2 + (phi1-phi2)**2)

    for _, event in dataframe_loose.iterrows():
        trijet_px = 0
        trijet_py = 0
        trijet_pz = 0
        trijet_E = 0
        for jet in event["trijet1stpass"]:
            trijet_px += jet[0]*math.cos(jet[2])
            trijet_py += jet[0]*math.sin(jet[2])
            trijet_pz += jet[0]*math.sinh(jet[1])
            trijet_E += jet[4]
        trijet_phi = math.atan(trijet_py/trijet_px)
        trijet_eta = math.atan(trijet_pz/trijet_E)
        
        angletoplep.append(deltaR(trijet_eta, trijet_phi, event["LeptonEta"], event["leptonphi"]))

    dataframe_loose["angletoplep"] = angletoplep

    angletop1top2 = []

    for _, event in dataframe_loose.iterrows():
        trijet1_px = 0
        trijet1_py = 0
        trijet1_pz = 0
        trijet1_E = 0
        trijet2_px = 0
        trijet2_py = 0
        trijet2_pz = 0
        trijet2_E = 0
        
        for jet in event["trijet1stpass"]:
            trijet1_px += jet[0]*math.cos(jet[2])
            trijet1_py += jet[0]*math.sin(jet[2])
            trijet1_pz += jet[0]*math.sinh(jet[1])
            trijet1_E += jet[4]
        trijet1_phi = math.atan(trijet1_py/trijet1_px)
        trijet1_eta = math.atan(trijet1_pz/trijet1_E)
        
        for jet in event["trijet2ndpass"]:
            trijet2_px += jet[0]*math.cos(jet[2])
            trijet2_py += jet[0]*math.sin(jet[2])
            trijet2_pz += jet[0]*math.sinh(jet[1])
            trijet2_E += jet[4]
        trijet2_phi = math.atan(trijet2_py/trijet2_px)
        trijet2_eta = math.atan(trijet2_pz/trijet2_E)
        
        angletop1top2.append(deltaR(trijet1_eta, trijet1_phi, trijet2_eta, trijet2_phi))

    dataframe_loose["angletop1top2"] = angletop1top2

    pTRat1st = []
    pTRat2nd = []

    for _, event in dataframe_loose.iterrows():
        px = 0
        py = 0
        HT = 0
        for jet in event["trijet1stpass"]:
            px += jet[0]*math.cos(jet[2])
            py += jet[0]*math.sin(jet[2])
            HT += jet[0]
        pTRat1st.append(math.sqrt(px**2 + py**2)/HT)
        
        px = 0
        py = 0
        HT = 0
        for jet in event["trijet2ndpass"]:
            px += jet[0]*math.cos(jet[2])
            py += jet[0]*math.sin(jet[2])
            HT += jet[0]
        pTRat2nd.append(math.sqrt(px**2 + py**2)/HT)

    dataframe_loose["pTRat1st"] = pTRat1st
    dataframe_loose["pTRat2nd"] = pTRat2nd

    topness = []
    ditopness = []
    tritopness = []

    for _, event in dataframe_loose.iterrows():
        topness.append(event["trijet1stpass"][0][5])
        ditopness.append(event["trijet2ndpass"][0][5])
        tritopness.append(event["trijet3rdpass"][0][5])
        
    dataframe_loose["topness"] = topness
    dataframe_loose["ditopness"] = ditopness
    dataframe_loose["tritopness"] = tritopness
    dataframe_loose.drop(columns=["trijet1stpass", "trijet2ndpass", "trijet3rdpass"], inplace=True)

add_variables(dataframe_loose)
dataframe_loose.to_pickle("fourtops_7J2M_pd.p")
print("Completed central samples")

print(delimiter)
print("Starting JESup samples")

df_sig_el_jesup = Toolset.to_pandas(sig_el_craneen_jesup, "Craneen__El", selection_el_loose, branchlist, branchlist_name, isSignal=True, add_weight=True)
df_bg_el_jesup  = Toolset.to_pandas(bg_el_craneen_jesup,  "Craneen__El", selection_el_loose, branchlist, branchlist_name, isSignal=False, add_weight=True)
df_sig_mu_jesup = Toolset.to_pandas(sig_mu_craneen_jesup, "Craneen__Mu", selection_mu_loose, branchlist, branchlist_name, isSignal=True, add_weight=True)
df_bg_mu_jesup  = Toolset.to_pandas(bg_mu_craneen_jesup,  "Craneen__Mu", selection_mu_loose, branchlist, branchlist_name, isSignal=False, add_weight=True)

for trijet in ["trijet1stpass", "trijet2ndpass", "trijet3rdpass"]:
    df_sig_el_jesup[trijet] = [np.asarray(row[0]) for row in root_numpy.root2array(sig_el_craneen_jesup, "Craneen__El", selection=selection_el_loose, branches=[trijet], step=1)]
    df_bg_el_jesup[trijet]  = [np.asarray(row[0]) for row in root_numpy.root2array(bg_el_craneen_jesup,  "Craneen__El", selection=selection_el_loose, branches=[trijet], step=1)]
    df_sig_mu_jesup[trijet] = [np.asarray(row[0]) for row in root_numpy.root2array(sig_mu_craneen_jesup, "Craneen__Mu", selection=selection_mu_loose, branches=[trijet], step=1)]
    df_bg_mu_jesup[trijet]  = [np.asarray(row[0]) for row in root_numpy.root2array(bg_mu_craneen_jesup,  "Craneen__Mu", selection=selection_mu_loose, branches=[trijet], step=1)]

df_sig_el_jesup["fromMu"] = 0
df_bg_el_jesup["fromMu"] = 0
df_sig_mu_jesup["fromMu"] = 1
df_bg_mu_jesup["fromMu"] = 1

dataframe_jesup = pd.concat([df_sig_el_jesup, df_bg_el_jesup, df_sig_mu_jesup, df_bg_mu_jesup], ignore_index=True)
#dataframe_jesup = pd.concat([df_bg_el_jesup, df_bg_mu_jesup], ignore_index=True)
#add_trijet_comb(dataframe_jesup)
dataframe_jesup["jes"] = 1

add_variables(dataframe_jesup)
dataframe_jesup.to_pickle("fourtops_7J2M_pd_jesup.p")
print("Completed JESup samples")

print(delimiter)
print("Starting JESdown samples")

df_sig_el_jesdown = Toolset.to_pandas(sig_el_craneen_jesdown, "Craneen__El", selection_el_loose, branchlist, branchlist_name, isSignal=True, add_weight=True)
df_bg_el_jesdown  = Toolset.to_pandas(bg_el_craneen_jesdown, "Craneen__El", selection_el_loose, branchlist, branchlist_name, isSignal=False, add_weight=True)
df_sig_mu_jesdown = Toolset.to_pandas(sig_mu_craneen_jesdown, "Craneen__Mu", selection_mu_loose, branchlist, branchlist_name, isSignal=True, add_weight=True)
df_bg_mu_jesdown  = Toolset.to_pandas(bg_mu_craneen_jesdown, "Craneen__Mu", selection_mu_loose, branchlist, branchlist_name, isSignal=False, add_weight=True)

for trijet in ["trijet1stpass", "trijet2ndpass", "trijet3rdpass"]:
    df_sig_el_jesdown[trijet] = [np.asarray(row[0]) for row in root_numpy.root2array(sig_el_craneen_jesdown, "Craneen__El", selection=selection_el_loose, branches=[trijet], step=1)]
    df_bg_el_jesdown[trijet]  = [np.asarray(row[0]) for row in root_numpy.root2array(bg_el_craneen_jesdown,  "Craneen__El", selection=selection_el_loose, branches=[trijet], step=1)]
    df_sig_mu_jesdown[trijet] = [np.asarray(row[0]) for row in root_numpy.root2array(sig_mu_craneen_jesdown, "Craneen__Mu", selection=selection_mu_loose, branches=[trijet], step=1)]
    df_bg_mu_jesdown[trijet]  = [np.asarray(row[0]) for row in root_numpy.root2array(bg_mu_craneen_jesdown,  "Craneen__Mu", selection=selection_mu_loose, branches=[trijet], step=1)]

df_sig_el_jesdown["fromMu"] = 0
df_bg_el_jesdown["fromMu"] = 0
df_sig_mu_jesdown["fromMu"] = 1
df_bg_mu_jesdown["fromMu"] = 1

dataframe_jesdown = pd.concat([df_sig_el_jesdown, df_bg_el_jesdown, df_sig_mu_jesdown, df_bg_mu_jesdown], ignore_index=True)
#dataframe_jesdown = pd.concat([df_bg_el_jesdown, df_bg_mu_jesdown], ignore_index=True)
#add_trijet_comb(dataframe_jesdown)
dataframe_jesdown["jes"] = -1

add_variables(dataframe_jesdown)
dataframe_jesdown.to_pickle("fourtops_7J2M_pd_jesdown.p")

dataframe_ultimate = pd.concat([dataframe_loose, dataframe_jesup, dataframe_jesdown])
dataframe_ultimate.to_pickle("fourtops_7J2M_pd_jes.p")
print("Completed JESdown samples")

# df_sig_el_jerup = Toolset.to_pandas(sig_el_craneen_jerup, "Craneen__El", selection_el_loose, branchlist, branchlist_name, isSignal=True, add_weight=True)
# df_bg_el_jerup  = Toolset.to_pandas(bg_el_craneen_jerup, "Craneen__El", selection_el_loose, branchlist, branchlist_name, isSignal=False, add_weight=True)
# df_sig_mu_jerup = Toolset.to_pandas(sig_mu_craneen_jerup, "Craneen__Mu", selection_mu_loose, branchlist, branchlist_name, isSignal=True, add_weight=True)
# df_bg_mu_jerup  = Toolset.to_pandas(bg_mu_craneen_jerup, "Craneen__Mu", selection_mu_loose, branchlist, branchlist_name, isSignal=False, add_weight=True)
# 
# df_sig_el_jerup["fromMu"] = 0
# df_bg_el_jerup["fromMu"] = 0
# df_sig_mu_jerup["fromMu"] = 1
# df_bg_mu_jerup["fromMu"] = 1
# 
# dataframe_jerup = pd.concat([df_sig_el_jerup, df_bg_el_jerup, df_sig_mu_jerup, df_bg_mu_jerup], ignore_index=True)
#dataframe_jerup = pd.concat([df_bg_el_jerup, df_bg_mu_jerup], ignore_index=True)
# 
# dataframe_jerup.to_pickle("fourtops_8J3M_pd_jerup.p")
# 
# df_sig_el_jerdown = Toolset.to_pandas(sig_el_craneen_jerdown, "Craneen__El", selection_el_loose, branchlist, branchlist_name, isSignal=True, add_weight=True)
# df_bg_el_jerdown  = Toolset.to_pandas(bg_el_craneen_jerdown, "Craneen__El", selection_el_loose, branchlist, branchlist_name, isSignal=False, add_weight=True)
# df_sig_mu_jerdown = Toolset.to_pandas(sig_mu_craneen_jerdown, "Craneen__Mu", selection_mu_loose, branchlist, branchlist_name, isSignal=True, add_weight=True)
# df_bg_mu_jerdown  = Toolset.to_pandas(bg_mu_craneen_jerdown, "Craneen__Mu", selection_mu_loose, branchlist, branchlist_name, isSignal=False, add_weight=True)
# 
# df_sig_el_jerdown["fromMu"] = 0
# df_bg_el_jerdown["fromMu"] = 0
# df_sig_mu_jerdown["fromMu"] = 1
# df_bg_mu_jerdown["fromMu"] = 1
# 
# dataframe_jerdown = pd.concat([df_sig_el_jerdown, df_bg_el_jerdown, df_sig_mu_jerdown, df_bg_mu_jerdown], ignore_index=True)
#dataframe_jerdown = pd.concat([df_bg_el_jerdown, df_bg_mu_jerdown], ignore_index=True)
# 
# dataframe_jerdown.to_pickle("fourtops_8J3M_pd_jerdown.p")
# 
# df_sig_el_jesupdiff = df_sig_el[["Eventnr", "multitopness"]].merge(df_sig_el_jesup[["Eventnr", "multitopness"]], on="Eventnr")
# df_sig_el_jesupdiff["jesup_diff"] = (df_sig_el_jesupdiff.multitopness_x - df_sig_el_jesupdiff.multitopness_y)
# df_sig_el_jesdowndiff = df_sig_el[["Eventnr", "multitopness"]].merge(df_sig_el_jesdown[["Eventnr", "multitopness"]], on="Eventnr")
# df_sig_el_jesdowndiff["jesdown_diff"] = (df_sig_el_jesdowndiff.multitopness_x - df_sig_el_jesdowndiff.multitopness_y)
# df_sig_el_withjes = df_sig_el.merge(df_sig_el_jesupdiff[["Eventnr", "jesup_diff"]], on="Eventnr").merge(df_sig_el_jesdowndiff[["Eventnr", "jesdown_diff"]], on="Eventnr")
# 
# df_bg_el_jesupdiff = df_bg_el[["Eventnr", "multitopness"]].merge(df_bg_el_jesup[["Eventnr", "multitopness"]], on="Eventnr")
# df_bg_el_jesupdiff["jesup_diff"] = (df_bg_el_jesupdiff.multitopness_x - df_bg_el_jesupdiff.multitopness_y)
# df_bg_el_jesdowndiff = df_bg_el[["Eventnr", "multitopness"]].merge(df_bg_el_jesdown[["Eventnr", "multitopness"]], on="Eventnr")
# df_bg_el_jesdowndiff["jesdown_diff"] = (df_bg_el_jesdowndiff.multitopness_x - df_bg_el_jesdowndiff.multitopness_y)
# df_bg_el_withjes = df_bg_el.merge(df_bg_el_jesupdiff[["Eventnr", "jesup_diff"]], on="Eventnr").merge(df_bg_el_jesdowndiff[["Eventnr", "jesdown_diff"]], on="Eventnr")
# 
# df_sig_mu_jesupdiff = df_sig_mu[["Eventnr", "multitopness"]].merge(df_sig_mu_jesup[["Eventnr", "multitopness"]], on="Eventnr")
# df_sig_mu_jesupdiff["jesup_diff"] = (df_sig_mu_jesupdiff.multitopness_x - df_sig_mu_jesupdiff.multitopness_y)
# df_sig_mu_jesdowndiff = df_sig_mu[["Eventnr", "multitopness"]].merge(df_sig_mu_jesdown[["Eventnr", "multitopness"]], on="Eventnr")
# df_sig_mu_jesdowndiff["jesdown_diff"] = (df_sig_mu_jesdowndiff.multitopness_x - df_sig_mu_jesdowndiff.multitopness_y)
# df_sig_mu_withjes = df_sig_mu.merge(df_sig_mu_jesupdiff[["Eventnr", "jesup_diff"]], on="Eventnr").merge(df_sig_mu_jesdowndiff[["Eventnr", "jesdown_diff"]], on="Eventnr")
# 
# df_bg_mu_jesupdiff = df_bg_mu[["Eventnr", "multitopness"]].merge(df_bg_mu_jesup[["Eventnr", "multitopness"]], on="Eventnr")
# df_bg_mu_jesupdiff["jesup_diff"] = (df_bg_mu_jesupdiff.multitopness_x - df_bg_mu_jesupdiff.multitopness_y)
# df_bg_mu_jesdowndiff = df_bg_mu[["Eventnr", "multitopness"]].merge(df_bg_mu_jesdown[["Eventnr", "multitopness"]], on="Eventnr")
# df_bg_mu_jesdowndiff["jesdown_diff"] = (df_bg_mu_jesdowndiff.multitopness_x - df_bg_mu_jesdowndiff.multitopness_y)
# df_bg_mu_withjes = df_bg_mu.merge(df_bg_mu_jesupdiff[["Eventnr", "jesup_diff"]], on="Eventnr").merge(df_bg_mu_jesdowndiff[["Eventnr", "jesdown_diff"]], on="Eventnr")
# 
# dataframe_withjes = pd.concat([df_sig_el_withjes, df_bg_el_withjes, df_sig_mu_withjes, df_bg_mu_withjes], ignore_index=True)
# dataframe_withjes.to_pickle("fourtops_8J3M_pd_withjes.p")