import cellrank as cr
import scanpy as sc
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import product
from scipy import stats
from sklearn.linear_model import LinearRegression
import statsmodels.stats.multitest as multitest
import projections as pj
import re
import scipy.spatial as sp, scipy.cluster.hierarchy as hc

# Load gene sets
df_em_list = pd.read_csv("EMTGeneUpdate3.txt", sep='\t| +') # EMT gene list
df_sclc_list = pd.read_csv('sig_matrix_parTI_2021.csv') # SCLC gene list
emt_set = set(df_em_list.Gene)
sclc_set = set(df_sclc_list['0'])
egenes_all = set(df_em_list[df_em_list.Annotation=='E'].Gene)
mgenes_all = set(df_em_list[df_em_list.Annotation=='M'].Gene)
# Alternative: Exclude SCLC genes from EMT genes
egenes_nosc = egenes_all - sclc_set
mgenes_nosc = mgenes_all - sclc_set
egenes, mgenes = egenes_nosc, mgenes_nosc
#egenes, mgenes = egenes_all, mgenes_all

# Convert human to mouse gene lists
df_m2h = pd.read_csv("M2H_genes.txt", sep=",")
mouse_egenes = set(df_m2h[df_m2h.HGNC.isin(egenes)].Gene)
mouse_mgenes = set(df_m2h[df_m2h.HGNC.isin(mgenes)].Gene)
mouse_emt_set = set(df_m2h[df_m2h.HGNC.isin(emt_set)].Gene)
mouse_sclc_set = set(df_m2h[df_m2h.HGNC.isin(sclc_set)].Gene)

##################### RPM Mouse scRNA-seq #############
##################### Figure 1A #######################
fname = './adata_04_by-timepoint_CR.h5ad'
adata = cr.read(fname)
adata_m = cr.read('./X_magic_02.h5ad') # For visualizing individual gene expression only
gene_names = adata.var.Accession.index
d_xx = adata.X # Using normalized, unimputed data for most analysis
#
# Run ssGSEA
resE = pj.ssgsea(d_xx, gene_names, mouse_egenes)
resM = pj.ssgsea(d_xx, gene_names, mouse_mgenes)
#
# Run nnPCA
pE, stdE, loadingsE = pj.nnpca(d_xx, gene_names, mouse_egenes)
pM, stdM, loadingsM = pj.nnpca(d_xx, gene_names, mouse_mgenes)
pcM, pcM2 = pj.rank_PCs(adata, pM)
pcE, pcE2 = pj.rank_PCs(adata, pE)
#
# Make a df for visualization.
projections = { 'E score (nnPC 1)':pcE,
                'M score (nnPC 1)':pcM,
                'E score (nnPC 2)':pcE2,
                'M score (nnPC 2)':pcM2,
                'E score (ssGSEA)':resE,
                'M score (ssGSEA)':resM,
            }
dfrpm = pj.make_emt_sclc_df(adata, adata_m.X, gene_names,
        mouse_emt_set|mouse_sclc_set,
        projections=projections)

fig, ax = pj.scatter_dist(df=dfrpm, vx = "E score (ssGSEA)",
        vy="M score (ssGSEA)", if_legend=True)
#pj.easy_save(fig, './figures/EM_RPM_legend.svg', dpi=600, fmt='svg')

#################### Figure 2A #############################
pj.scatter_dist(df=dfrpm, vx = "Cdh1", vy="Vim")

#################### Figure 2E #############################
pj.scatter_dist(df=dfrpm, vx = "Cdh1", vy="Zeb1")

fig, ax = pj.scatter_dist(df=dfrpm, vx = "Zeb1", vy="Vim")



#################### Figure 2C #############################
pj.scatter_dist(df=dfrpm, vx = "E score (nnPC 1)",
        vy="M score (nnPC 1)")

pj.scatter_dist(df=dfrpm, vx = "E score (nnPC 1)",
        vy="M score (nnPC 2)")



################## Bulk RNA-seq, Human Cell Lines #########
################## Figure 1B ##############################
data = pd.read_csv('./SCLC_combined_Minna_CCLE_batch_corrected_wo_lowgenes.csv', header = 0, index_col=0).T
gene_names = data.columns
resE = pj.ssgsea(data, gene_names, egenes)
resM = pj.ssgsea(data, gene_names, mgenes)
clines = pd.read_csv("./combined_clusters_2020-05-27-MC copy.csv", index_col = 0)
clus = clines.reindex(data.index).NEW_10_2020
dfbk = pj.make_emt_sclc_df(None, data, gene_names,
        emt_set|sclc_set, meta_data={'Subtype': clus},
        projections={'E score (ssGSEA)':resE.values,'M score (ssGSEA)':resM.values})

fig, ax = pj.scatter_dist(df=dfbk, vx = "E score (ssGSEA)", vy="M score (ssGSEA)",
        thresh=0.35, if_show=False, if_legend=True)
ax.set_ylim(-0.5, 0.61) # GSEA
#pj.easy_save(fig, './figures/EM_clines.svg', dpi=600, fmt='svg')
plt.show()


##################### 8 Human Cell Line scRNA-seq #####
##################### Figure 1A #######################
adata = cr.read('./adata_03b.h5ad') # Normalized data
adata_m = cr.read('./X_magic_03b.h5ad') # Imputed data
gene_names = adata.var.Accession.index
d_xx = adata.X # Using normalized, unimputed data for most analysis
#
# Run ssGSEA
#resE = pj.ssgsea(d_xx, gene_names, egenes_nosc)
#resM = pj.ssgsea(d_xx, gene_names, mgenes_nosc)
#
# Run nnPCA
pE, stdE, loadingsE = pj.nnpca(d_xx, gene_names, egenes_nosc)
pM, stdM, loadingsM = pj.nnpca(d_xx, gene_names, mgenes_nosc)
pcM, pcM2 = pj.rank_PCs(adata, pM)
pcE, pcE2 = pj.rank_PCs(adata, pE)
#
# Make a df for visualization.
projections = { 'E score (nnPC 1)':pcE,
                'M score (nnPC 1)':pcM,
                'E score (nnPC 2)':pcE2,
                'M score (nnPC 2)':pcM2,
                #'E score (ssGSEA)':resE,
                #'M score (ssGSEA)':resM,
}
df8cls = pj.make_emt_sclc_df(adata, adata_m.X, gene_names,
        emt_set|sclc_set,
        projections=projections)

#################### Figure 2B #############################
pj.scatter_dist(df=df8cls, vx = "CDH1", vy="VIM")

#################### Figure 2F #############################
fig, ax = pj.scatter_dist(df=df8cls, vx = "CDH1", vy="ZEB1", if_show=False)
ax.set_ylim(-0.3, 0.8)
#pj.easy_save(fig, './figures/scatter_scCellLines_CDH1_ZEB1.png', fmt='png', dpi=600)

fig, ax = pj.scatter_dist(df=df8cls, vx="ZEB1", vy="VIM", if_save=False)

#################### Figure 2D #############################
pj.scatter_dist(df=df8cls, vx = "E score (nnPC 1)",
        vy="M score (nnPC 1)")

pj.scatter_dist(df=df8cls, vx = "E score (nnPC 1)",
        vy="M score (nnPC 2)")


######## Classify M genes based on RPM and sc cell line data###

mmg = {'Vim'} | mouse_mgenes
hmg = set(df_m2h[df_m2h.Gene.isin(mmg)].HGNC)
datN_rpm = dfrpm.loc[dfrpm.Subtype=='A/N',dfrpm.columns.isin(mmg)]
datY_rpm = dfrpm.loc[dfrpm.Subtype=='Y',dfrpm.columns.isin(mmg)]
pvs_rpm = stats.ttest_ind(datN_rpm, datY_rpm)[1]
dfm_rpm = dfrpm.loc[:,dfrpm.columns.isin(list(mmg)+['Subtype'])].groupby('Subtype').agg('mean')
dfm_rpm = dfm_rpm.dropna().T
dfm_rpm.columns = [ 'RPM '+c for c in dfm_rpm]
dfm_rpm['Mean(N)-Mean(Y) (RPM)'] = dfm_rpm['RPM A/N'] - dfm_rpm['RPM Y']
dfm_rpm['FDR (RPM)'] = multitest.fdrcorrection(pvs_rpm)[1]
dfm_rpm
datN_h = df8cls.loc[df8cls.Subtype=='N',df8cls.columns.isin(hmg)]
datY_h = df8cls.loc[df8cls.Subtype=='Y',df8cls.columns.isin(hmg)]
pvs_h = stats.ttest_ind(datN_h, datY_h)[1]
dfm_h = df8cls.loc[:,df8cls.columns.isin(list(hmg)+['Subtype'])].groupby('Subtype').agg('mean')
dfm_h.columns = dfm_h.columns.map(dict(zip(df_m2h.HGNC, df_m2h.Gene)))
dfm_h = dfm_h.dropna().T
dfm_h.columns = [ 'Cell line '+c for c in dfm_h]
dfm_h['Mean(N)-Mean(Y) (Cell line)'] = dfm_h['Cell line N'] - dfm_h['Cell line Y']
dfm_h['FDR (Cell line)'] = multitest.fdrcorrection(pvs_h)[1]
dfm_h
dfm = pd.concat([dfm_rpm, dfm_h], axis=1, join='inner')


############ Scatter plot for M1 and M2 genes ##############
fig, ax = plt.subplots(figsize=(4,4))
fig.subplots_adjust(left=0.2, bottom=0.2)
xname, yname = 'Mean(N)-Mean(Y) (RPM)', 'Mean(N)-Mean(Y) (Cell line)'
sns.regplot(x=xname, y=yname, data=dfm,
        scatter=True, truncate=False, scatter_kws={'color':'gray'},
        )
for g in ('Zeb1', 'Twist1', 'Vim'):
    x, y = dfm.loc[g,xname], dfm.loc[g,yname]
    ax.text(x-0.4, y+0.1, g.upper())
    ax.scatter([x], [y], c='r')
ax.axvline(x=0, c='k', lw=1, zorder=-15)
ax.axhline(y=0, c='k', lw=1, zorder=-15)
#pj.easy_save(fig, './figures/M12_RPM_CL_scatter.svg', fmt='svg', dpi=600)
plt.show()


############ Heatmap for M1 and M2 genes ##################
mg1 = dfm[(dfm['RPM A/N']-dfm['RPM Y']>0)&
        (dfm['Cell line N']-dfm['Cell line Y']>0)&
        (dfm['FDR (RPM)']<0.05)&
        (dfm['FDR (Cell line)']<0.05)].index
mg2 = dfm[(dfm['RPM A/N']-dfm['RPM Y']<0)&
        (dfm['Cell line N']-dfm['Cell line Y']<0)&
        (dfm['FDR (RPM)']<0.05)&
        (dfm['FDR (Cell line)']<0.05)].index
dfcd = dfm.loc[:,dfm.columns.str.match('(RPM (A/N|Y))|(Cell line (N|Y))')]
mg_pal = sns.husl_palette(3)
mg_cs = {}
for g in dfcd.index:
    if g in mg1:
        mg_cs[g] = mg_pal[0]
    elif g in mg2:
        mg_cs[g] = mg_pal[1]
mg_cs = pd.Series(mg_cs)
#linkage = hc.linkage(dfcd.loc[mg1.tolist() + mg2.tolist(), ['RPM Y']], 'average')
#linkage = hc.linkage(dfcd.loc[mg1.tolist() + mg2.tolist(), ['RPM A/N', 'RPM Y']], 'average')
#linkage = hc.linkage(dfcd.loc[mg1.tolist() + mg2.tolist(), ['RPM A/N', 'RPM Y']], 'median')
#linkage = hc.linkage(dfcd.loc[mg1.tolist() + mg2.tolist(), ['Cell line N', 'Cell line Y']], 'ward')
linkage = hc.linkage(dfcd.loc[mg1.tolist() + mg2.tolist(), ['Cell line N', 'Cell line Y', 'RPM A/N', 'RPM Y']], 'median')
g = sns.clustermap(dfcd.loc[mg1.tolist() + mg2.tolist(), :].iloc[:,:4].T, center=0, cmap="vlag",
    col_colors=mg_cs,
    col_linkage = linkage,
    row_cluster=False,
    xticklabels=True,
    vmin=-0.1, vmax=0.1,
    dendrogram_ratio=(.0, 0.2),
    cbar_pos=(.92, .85, .01, .13),
    linewidths=.75, figsize=(15, 6))
g.ax_cbar.set_yticklabels([r'$\leq$-1',0, r'$\geq$1'])
g.ax_cbar.set_ylabel('Normalized\nexpression')
plt.setp(g.ax_heatmap.xaxis.get_majorticklabels(), rotation=70, size=10)
for tick_label in g.ax_heatmap.axes.get_xticklabels():
    tick_text = tick_label.get_text()
    if tick_text in ['Zeb1', 'Zeb2', 'Fn1', 'Vim', 'Twist1']:
        tick_label.set_color('r')
    tick_label.set_text(tick_text.upper())
#pj.easy_save(g.fig, './figures/clustermap.svg', fmt='svg', dpi=600)
plt.show()

