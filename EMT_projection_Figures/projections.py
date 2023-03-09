from subprocess import check_output
import sys
import os.path
import re
import numpy as np
import scipy as sp
import pandas as pd
import gseapy as gp
import seaborn as sns
import colorcet as cc
import matplotlib.pyplot as plt
# Set up nnPCA. Not needed for nnGSEA
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
pandas2ri.activate()
r = ro.r
r['source']('nnpca.R')# Loading the function we have defined in R.
nnpca_r = ro.globalenv['nnpca_r']# nnPCA function in python

cset = sns.color_palette("husl", 8)
cset2 = sns.color_palette("Spectral", 6)
palette_default = {
    'A':cset[0],
    'A (2021)':cset[0], 'A/N': cset[-1], 'A2': cset[1], 'N (2021)': cset[-2],
    'P/Y': cset[3],'P (2021)':cset[2], 'Y (2021)': cset[5], 'Generalist': 'lightgray',
    'New A':cset[0], 'New A2': cset[1], 'N': cset[-2],'P':cset[2], 'Y': cset[5], 'Generalist': 'lightgray',
    '4': cset2[0], '7': cset2[1], '11': cset2[2],
    '14': cset2[3], '17': cset2[4], '21': cset2[5], 'None':(1,1,1,1)}

#patient_pal = sns.color_palette(cc.glasbey, n_colors=len(dfpc.Patient.unique()))
#for i, patient in enumerate(list(dfpc.Patient.unique())):
    #palette.update({patient: patient_pal[i]})
#treatment_pal = sns.color_palette(cc.glasbey, n_colors=len(dfpc.Treatment.unique()))
#for i, treatment in enumerate(list(dfpc.treatment.unique())):
    #palette.update({treatment: treatment_pal[i]})

def update_pal(pal, data):
    '''
    Add colors to pal based on values of data. Color map is
    cc.glasbey.
    pal: palette dictionary
    data: Pandas series
    '''
    new_pal = sns.color_palette(cc.glasbey, n_colors=len(data.unique()))
    for i, cat in enumerate(list(data.unique())):
        pal.update({cat: new_pal[i]})


def nnpca(data, gene_names_all, gene_names_in_set, n_components=5):
    '''
    Compute ssGESA score for a gene set in expression data
    data: gene expression matrix. Rows: samples/cells.  Columns: genes
    gene_names_all: names of all genes in the dataset (Any array-like type)
    gene_names_in_set: names of genes in the gene set (Set or List)
    '''
    if isinstance(data, sp.sparse._csr.csr_matrix):
        data = data.toarray()
    elif isinstance(data, pd.core.frame.DataFrame):
        data = data.values
    if_in_set = gene_names_all.isin(gene_names_in_set)
    d_sub = data[:, if_in_set]
    res =  nnpca_r(d_sub, n_components)
    stds, loadings, projections = res[0], res[1], res[-1]
    return projections, stds, loadings

def ssgsea(data, gene_names_all, gene_names_in_set, name='GeneSet'):
    '''
    Compute ssGESA score for a gene set in expression data
    data: gene expression matrix. Rows: samples/cells.  Columns: genes
    gene_names_all: names of all genes in the dataset (Any array-like type)
    gene_names_in_set: names of genes in the gene set (Set or List)
    '''
    gs_dict = {name: list(gene_names_in_set)}
    if isinstance(data, sp.sparse._csr.csr_matrix):
        data = data.T.toarray()
    elif isinstance(data, np.ndarray):
        data = data.T
    elif isinstance(data, pd.DataFrame):
        data = data.T
    df = pd.DataFrame(data, index=gene_names_all)
    ssgs = gp.ssgsea(data=df,
        gene_sets=gs_dict,
        outdir=None,
        sample_norm_method='rank', # choose 'custom' for your own rank list
        no_plot=True,
        threads=8)
    return ssgs.res2d.set_index('Name').NES

def geti(gene_names_all, gene):
    '''
    Get positional index of a gene from a gene list
    gene_names_all: list of gene symbols
    gene: target gene symbol
    '''
    return pd.Index(gene_names_all).get_loc(gene)

def getge(d, gene_names_all, gene):
    '''
    Get expression vector for a gene
    d: expression
    gene: A gene symbol string
    '''
    if isinstance(d, sp.sparse._csr.csr_matrix):
        return d[:,geti(gene_names_all, gene)].toarray().flatten().astype('float64')
    if isinstance(d, np.ndarray):
        return d[:,geti(gene_names_all, gene)].flatten().astype('float64')

def make_emt_sclc_df(adata, expr_data, gene_names_all,
        gene_names, projections={}, meta_data={}, pal={}):
    '''
    Construct a dataframe for key genes, metadata and projections
    for visualization.
    adata: AnnData object for extracting sample info.
    expr_data: Expression matrix (sparse, numpy array, pandas dataframe).
    gene_names_all: List of gene symbols.
    gene_names: Gene set.
    projections: Names/Values dictionary for projections
    meta_data: sample information to add.
    pal: colors to add.
    '''
    gids = []
    gene_names_found = []
    m_not_found = []
    for g in gene_names:
        if g in gene_names_all:
            gids.append(geti(gene_names_all, g))
            gene_names_found.append(g)
        else:
            m_not_found.append(g)
    print('Genes not found when making visualization DataFrame: ', m_not_found)
    if isinstance(expr_data, sp.sparse._csr.csr_matrix):
        d = expr_data[:,gids].toarray().astype('float64')
    if isinstance(expr_data, np.ndarray):
        d = expr_data[:,gids].astype('float64')
    if isinstance(expr_data, pd.DataFrame):
        d = expr_data.iloc[:,gids].astype('float64')
    print(d.shape, len(gene_names_found))
    df = pd.DataFrame(data=d, columns=gene_names_found)

    for k, v in projections.items():
        df[k] = v
    for k, v in meta_data.items():
        df[k] = v
    default_keys = {'SCLC-A_Score_pos': 'A score',
                    'SCLC-A2_Score_pos': 'A2 score',
                    'SCLC-N_Score_pos': 'N score',
                    'SCLC-P_Score_pos': 'P score',
                    'SCLC-Y_Score_pos': 'Y score',
                    'treatment': 'Treatment',
                    'arc_aa_type': 'Subtype',
                    'cline': 'Cell line',
                    'patient': 'Patient',
                    'SCLC_subtype': 'Subtype',
                    'ParetoTI_specialists': 'Subtype'
                    }
    for k, v in default_keys.items():
        if adata is None:
            break
        if k in adata.obs and not k in meta_data and not k in projections:
            df[v] = adata.obs[k].values

    if 'Subtype' in df.columns:
        df.Subtype.replace({'SCLC-A Specialist': 'A',
                            'SCLC-A2 Specialist': 'A2',
                            'SCLC-N Specialist': 'N',
                            'SCLC-P Specialist': 'P',
                            'SCLC-Y Specialist': 'Y'}, inplace=True)
        df.Subtype.replace({'SCLC-A': 'A',
                            'SCLC-A2': 'A2',
                            'SCLC-N': 'N',
                            'SCLC-P': 'P',
                            'SCLC-Y': 'Y'}, inplace=True)
        if df.Subtype.dtype.name == 'category' and not 'Generalist' in df.Subtype.cat.categories:
            df.Subtype.cat.add_categories('Generalist', inplace=True)
            df.Subtype.fillna(value='Generalist', inplace=True)
        df = df[df.Subtype != 'Unclassified']
        df = df[df.Subtype != 'Generalist']
        df = df[df.Subtype != 'None']

    # update palette
    palette = palette_default
    for k, v in pal.items():
        palette.update({k:v})
    if 'Treatment' in df.columns:
        treatment_pal = sns.color_palette(cc.glasbey, n_colors=len(df.Treatment.unique()))
        for i, treatment in enumerate(list(df.Treatment.unique())):
            palette.update({treatment: treatment_pal[i]})
    if 'Patient' in df.columns:
        patient_pal = sns.color_palette(cc.glasbey, n_colors=len(df.Patient.unique()))
        for i, patient in enumerate(list(df.Patient.unique())):
            palette.update({patient: patient_pal[i]})
    return df

def rank_PCs(adata, pcs, rank_by_subtypes=True):
    '''
    Rank PCs based on subtype std, and return top 2 PCs.
    adata: AnnData object.
    pcs: multiple PCs in 2D array. Each column is a PC.
    '''
    if rank_by_subtypes == False:
        return pcs[:,0], pcs[:,1]
    elif 'ParetoTI_specialists' in adata.obs.columns:
        subtypes = adata.obs.ParetoTI_specialists
        if subtypes.dtype.name == 'category' and not 'Generalist' in subtypes.cat.categories:
            subtypes.cat.add_categories('Generalist', inplace=True)
        subtypes.fillna(value='Generalist', inplace=True)
        #print(subtypes)
        dfm = pd.DataFrame(pcs)
        dfm['Subtype'] = subtypes.values
        dfpc = dfm.groupby('Subtype').agg('mean')
        stds = dfpc[dfpc.index!='Generalist'].std()
        print(stds)
        pco = stds.sort_values(ascending=False).index
        pc1, pc2 = dfm.iloc[:,pco[0]].values, dfm.iloc[:,pco[1]].values
        return pc1, pc2
    elif 'arc_aa_type' in adata.obs.columns:
        subtypes = adata.obs.arc_aa_type
        if subtypes.dtype.name == 'category' and not 'Generalist' in subtypes.cat.categories:
            subtypes.cat.add_categories('Generalist', inplace=True)
        subtypes.fillna(value='Generalist', inplace=True)
        subtypes.replace({'SCLC-A Specialist': 'A',
                            'SCLC-A2 Specialist': 'A2',
                            'SCLC-N Specialist': 'N',
                            'SCLC-P Specialist': 'P',
                            'SCLC-Y Specialist': 'Y'}, inplace=True)
        subtypes.replace({'Unclassified': 'Generalist'}, inplace=True)
        dfm = pd.DataFrame(pcs)
        dfm['Subtype'] = subtypes.values
        dfpc = dfm.groupby('Subtype').agg('mean')
        stds = dfpc[dfpc.index!='Generalist'].std()
        print(stds)
        pco = stds.sort_values(ascending=False).index
        pc1, pc2 = dfm.iloc[:,pco[0]].values, dfm.iloc[:,pco[1]].values
        return pc1, pc2
    else:
        print('Key for subtypes not found when ranking PCs.')
        return pcs[:,0], pcs[:,1]

def find_valid_file_name(file_name):
    '''
    Find a file name that does not exist in a directory
    '''
    sys.setrecursionlimit(20000)
    if os.path.isfile(file_name):
        if re.search('\.', file_name):
            m = re.search("\((\d+)\)\.", file_name)
            if m is not None:
                idx = str(int(m.group(1)) + 1)
                file_name_new = re.sub("\d+(?=\)\.)", idx, file_name)
            else:
                file_name_new = re.sub("(\.\w+)", r'(1)\1', file_name)
            file_name_new = find_valid_file_name(file_name_new)
            return file_name_new
        else:
            m = re.search("\((\d+)\)", file_name)
            if m is not None:
                idx = str(int(m.group(1)) + 1)
                file_name_new = re.sub("\d+(?=\))", idx, file_name)
            else:
                file_name_new = file_name + '(1)'
            file_name_new = find_valid_file_name(file_name_new)
            return file_name_new
    else:
        return file_name

def easy_save(fig, fig_name, dpi=300, fmt='png'):
    '''
    Save a figure object to a file. A non-redundant file name
    is created.
    '''
    if re.search("^\/\w*\/", fig_name):
        file_name = fig_name
    else:
        cwd = check_output(["pwd"]).decode().rstrip()
        if re.search("^\.\/", fig_name):
            file_name = fig_name[2:]
        file_name = cwd + '/' + file_name
    if re.search("\.\w+$", file_name):
        pass
    else:
        file_name = file_name + '.' + fmt
    file_name = find_valid_file_name(file_name)
    print("Saving figure to", file_name, fmt)
    fig.savefig(file_name, format=fmt, dpi=dpi)
    return file_name

def plot_dist(df, atype, label="Subtype",
        x='E score', y='M score', palette=None,
        ax=None, thresh=0.25, alpha=0.3, lw=2):
    '''
    Make 2D error bars and means.
    df: DataFrame object
    atype: Label value to select
    label: Label type (column) to select
    x, y: Columns to visualize
    ax: Axes object
    thresh: threshold of contour plot.
    '''
    if palette == None:
        palette = palette_default
    dsel = df[df[label].isin([atype])]
    sns.kdeplot(x=x, y=y, data=dsel, hue=label, ax=ax, fill=True, alpha=alpha, thresh=thresh, palette=palette, legend=False)
    ax.errorbar(dsel[x].mean(), dsel[y].mean(),
        xerr=dsel[x].std(), yerr=dsel[y].std(),
        ecolor='k', mfc=palette[atype], mec='w', ms=6, mew=1,
        #fmt='o', label=atype, zorder=10, lw=lw)
        fmt='o', label='_nolegend_', zorder=10, lw=lw)

def scatter_dist(df, cg="Subtype", vx="E score (nnPC 1)",
        vy='M score (nnPC 1)', if_save=False, if_show=True,
        thresh=0.02, if_legend=False,
        size=1, alpha=0.2, fname='', palette=None, **kwargs):
    '''
    Make 2D distribution plot. Use given column (cg) for hue (categorical).
    df: DataFrame object
    cg: Label type (column) to use as hue (categorical)
    vx, vy: Columns to visualize
    thresh: threshold of contour plot.
    '''
    df2p = df
    if palette == None:
        palette = palette_default
    if "A/N" in df.Subtype.values:
        hue_order = ["A/N", "A2", "Y", 'P/Y']
    elif 'A' in df.Subtype.values:
        hue_order = ['A', 'A2', 'N', 'P', 'Y']
    fig, ax = plt.subplots(figsize=(5.3,4))
    fig.subplots_adjust(left=0.15, bottom=0.135, right=0.7)
    if cg == 'Subtype':
        sns.scatterplot(x=vx, y=vy, data=df2p.sort_values('Subtype', key=np.vectorize(hue_order.index)), hue=cg, ax=ax, alpha=alpha, size=1, hue_order=hue_order, legend=if_legend, palette=palette)
    else:
        sns.scatterplot(x=vx, y=vy, data=df2p, hue=cg, ax=ax, alpha=alpha, size=1, legend=if_legend, palette=palette)
    for cat in df[cg].unique():
        plot_dist(df2p, cat, label=cg, x=vx, y=vy, ax=ax, thresh=thresh, alpha=0.2)
    if if_save:
        easy_save(fig, './figures/projections_'+
            fname.replace('./', '').replace('/', '_').replace('.h5ad', '')
            + '_' + cg.replace(' ', '_') + '_' +
            vx.replace(' ', '_') + '_' + vy.replace(' ', '_') + '.png', fmt='png', dpi=600)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(12)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(12)
    ax.xaxis.label.set_size(12)
    ax.yaxis.label.set_size(12)
    if if_show:
        plt.show()
    return fig, ax

def scatter_score(df, cg="Y score", vx="E score (nnPC 1)",
        vy='M score (nnPC 1)', if_save=False, if_show=True,
        size=1, alpha=0.2, fname='', **kwargs):
    '''
    Make 2D distribution plot. Use given column (cg) for hue (coninuous).
    df: DataFrame object
    cg: Label type (column) to use as hue (continuous)
    vx, vy: Columns to visualize
    '''
    df2p = df
    if 'vmin' in kwargs:
        cmin = kwargs['vmin']
    else:
        cmin = df2p[cg].min()
    if 'vmax' in kwargs:
        cmax = kwargs['vmax']
    else:
        cmax = df2p[cg].max()
    norm = plt.Normalize(cmin, cmax)
    #
    fig, ax = plt.subplots(figsize=(3.3,3))
    fig.subplots_adjust(left=0.15, bottom=0.235, right=0.7)
    sns.scatterplot(x=vx, y=vy, data=df2p, hue=cg, hue_norm = norm, ax=ax,
        size=size, alpha=alpha,legend=True, palette='rocket_r', **kwargs)
    #
    sm = plt.cm.ScalarMappable(cmap="rocket_r", norm=norm)
    sm.set_array([])
    # Remove the legend and add a colorbar
    ax.get_legend().remove()
    cax = fig.add_axes([0.8, 0.2, 0.03, 0.60])
    fig.colorbar(sm, cax=cax)
    cax.set_ylabel(cg)
    if if_save:
        easy_save(fig, './figures/nnPCA_SubtypeScore_'+
            fname.replace('./', '').replace('/', '_').replace('.h5ad', '')
            + '_' + cg.replace(' ', '_') + '_' +
            vx.replace(' ', '_') + '_' + vy.replace(' ', '_') + '.png', fmt='png', dpi=600)
    if if_show:
        plt.show()

