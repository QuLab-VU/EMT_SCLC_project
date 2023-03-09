import cellrank as cr
import scanpy as sc
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import product
from scipy import stats
import projections as pj
import re

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
egenes, mgenes = egenes_all, mgenes_all

# Load datasets (SC53, SC68)
fnames = ['./mda/SC53.h5ad', './mda/SC68.h5ad']
dfs = []
for fname in fnames:
    adata = cr.read(fname)
    gene_names = adata.var.Accession.index
    d_xx = adata.X # Using normalized, unimputed data for most analysis
    #
    # Run ssGSEA
    #resE = pj.ssgsea(d_xx, gene_names, egenes)
    #resM = pj.ssgsea(d_xx, gene_names, mgenes)
    #
    # Run nnPCA
    pE, stdE, loadingsE = pj.nnpca(d_xx, gene_names, egenes)
    pM, stdM, loadingsM = pj.nnpca(d_xx, gene_names, mgenes)
    pcE, pcE2, pcM, pcM2 = pE[:,0], pE[:,1], pM[:,0], pM[:,1]
    #
    # Make a df for visualization.
    #import projections as pj
    projections = { 'E score (nnPC 1)':pcE,
                    'M score (nnPC 1)':pcM,
                    'E score (nnPC 2)':pcE2,
                    'M score (nnPC 2)':pcM2,
                }
    dfpc = pj.make_emt_sclc_df(adata, d_xx, gene_names,
            emt_set|sclc_set,
            projections=projections)
    dfs.append(dfpc)
df53, df68 = dfs

##### Human tumors from MSK study ########################
adata_a2 = sc.read_h5ad('./HTAN_w_A2scores.h5ad')
d_xx = adata_a2.layers['imputed_normalized']
gene_names = adata_a2.var.index
# Run nnPCA
pE, stdE, loadingsE = pj.nnpca(d_xx, gene_names, egenes)
pM, stdM, loadingsM = pj.nnpca(d_xx, gene_names, mgenes)
pcE, pcE2, pcM, pcM2 = pE[:,0], pE[:,1], pM[:,0], pM[:,1]
#
# Make a df for visualization.
#import projections as pj
projections = { 'E score (nnPC 1)':pcE,
                'M score (nnPC 1)':pcM,
                'E score (nnPC 2)':pcE2,
                'M score (nnPC 2)':pcM2,
            }
dfmsk = pj.make_emt_sclc_df(adata_a2, d_xx, gene_names,
        emt_set|sclc_set,
        projections=projections)
###### Create 'Cluster' column for new subtyping results
dfmsk['Cluster'] = dfmsk['Subtype']
dfmsk.Cluster.replace({'A': 'New A'}, inplace=True)
dfmsk['Cluster'] = dfmsk['Cluster'].cat.add_categories('New A2')
dfmsk['Cluster'][dfmsk['M score (nnPC 1)'] < (dfmsk['E score (nnPC 1)']*0.05-1.7)] = 'New A2'
dfs.append(dfmsk)


# Single-patient RU1108 data from the MSK set
adata_ru = adata_a2[np.where(dfmsk.Patient=='RU1108')[0],:]
d_xx_ru = adata_ru.X
pE, stdE, loadingsE = pj.nnpca(d_xx_ru, gene_names, egenes)
pM, stdM, loadingsM = pj.nnpca(d_xx_ru, gene_names, mgenes)
pcE, pcE2, pcM, pcM2 = pE[:,0], pE[:,1], pM[:,0], pM[:,1]
#
# Make a df for visualization.
#import projections as pj
projections = { 'E score (nnPC 1)':pcE,
                'M score (nnPC 1)':pcM,
                'E score (nnPC 2)':pcE2,
                'M score (nnPC 2)':pcM2,
            }
dfru = pj.make_emt_sclc_df(adata_ru, d_xx_ru, gene_names,
        emt_set|sclc_set,
        projections=projections)
dfs.append(dfru)


################# Visualize MSK dataset #################
pj.scatter_dist(df=dfmsk, cg='Patient', thresh=0.7, alpha=0.05,
        vx='E score (nnPC 1)', vy='M score (nnPC 1)')
plt.show()

pj.scatter_dist(df=dfmsk, cg='Treatment', thresh=0.7, alpha=0.05,
        vx='E score (nnPC 1)', vy='M score (nnPC 1)', if_legend=True)
plt.show()

pj.scatter_dist(df=dfmsk, alpha=0.05,
        vx='E score (nnPC 1)', vy='M score (nnPC 1)')
plt.show()

pj.scatter_dist(df=dfmsk, cg='Patient', thresh=0.7, alpha=0.05,
        vx='ZEB1', vy='VIM')
plt.show()

pj.scatter_dist(df=dfmsk, cg='Subtype', thresh=0.7, alpha=0.05,
        vx='ZEB1', vy='VIM')
plt.show()


################# Boxplots for scores (MSK) ################
vys = ['E score (nnPC 1)', 'A2 score', 'A score', 'N score', 'P score', 'Y score']
clus_order = ['New A2', 'New A', 'N', 'P']
for vy in vys:
    fig, ax = plt.subplots(figsize=(2.5,2))
    fig.subplots_adjust(left=0.3, bottom=0.3)
    sns.boxplot(x='Cluster', y=vy, data=dfmsk.sort_values('Cluster', key=np.vectorize(clus_order.index))
        ,order=clus_order, hue_order='Cluster', ax=ax, palette=palette)
    ax.set_xticklabels(['A2*', 'A*', 'N', 'P'])
    #ax.set_xlabel('Expression')
    #pj.easy_save(fig, './figures/ScoreBox_MSK_'+vy.replace(' ', '_')+'.svg', dpi=600, fmt='svg')
plt.show()


################# SCLC scores on EM projections (MSK) ######
scores = ('A2 score','A score', 'N score', 'P score', 'Y score')
proj= ( ('E score (nnPC 1)', 'M score (nnPC 1)'),
        )
for cg, vps in product(scores, proj):
    vx, vy = vps
    pj.scatter_score(dfmsk, fname='MSK', cg=cg, vx=vx, vy=vy, alpha=0.3)
plt.show()#



################# SCLC scores on EM projections (SC53) ######
scores = ('A2 score','A score', 'N score', 'P score', 'Y score')
proj= (#('E score (ssGSEA)', 'M score (ssGSEA)'),
        ('E score (nnPC 1)', 'M score (nnPC 1)'),
        #('E score (nnPC 1)', 'M score (nnPC 2)')
        )
for cg, vps in product(scores, proj):
    vx, vy = vps
    pj.scatter_score(df53, fname='SC53', cg=cg, vx=vx, vy=vy, alpha=0.3, if_save=True, if_show=False, vmin=0, vmax=0.8)
plt.show()#


########## Make a DataFrame for 3 tumors ###############
dfs2p = [dfru, df53, df68]
tumors = ['RU1108', 'SC53', 'SC68']
g1s = ['E score', 'CDH1 expression']
samp_types = ['All cells', 'ASCL1+ cells']
g2s = ['A2 score', 'A score', 'N score', 'P score', 'Y score']
d2p = []
for g2, samp_type in product(g2s, samp_types):
    row_data = []
    for tumor, g1 in product(tumors, g1s):
        df = dfs2p[tumors.index(tumor)].dropna()
        if samp_type == 'ASCL1+ cells':
            df = df[df.ASCL1>0.0]
        g1 = g1.replace(' expression', '').replace(' score', ' score (nnPC 1)')
        corr, p = stats.spearmanr(df[g1], df[g2])
        if p > 0.05:
            print(g1, g2, tumor, samp_type)
        row_data.append(corr)
    d2p.append(row_data)
index = pd.MultiIndex.from_product((g2s, samp_types), names=('Score 2', 'Cell type'))
cols = pd.MultiIndex.from_product((tumors, g1s), names=('Tumor', 'Score 1'))
df2p = pd.DataFrame(d2p, index=index, columns=cols)


############# Heatmap of Spearman r (3 tumors) ###########
lut = dict(zip(tumors, "rbg"))
lut_rows = dict(zip(g2s, sns.color_palette("husl", 5)))
col_colors = df2p.columns.get_level_values('Tumor').map(lut)
row_colors = df2p.index.get_level_values('Score 2').map(lut_rows)
g = sns.clustermap(df2p, col_cluster=False, row_cluster=False,
        center=0, cmap='RdBu_r', figsize=(5,6), linewidth=0.5,
        col_colors=col_colors, row_colors=row_colors,
        cbar_kws={'label': r'Spearman $\it{r}$', 'orientation': 'horizontal'},
        cbar_pos=(0.6, 0.94, 0.18, 0.03),
        dendrogram_ratio=(0.25, 0.2))
axh = g.ax_heatmap
for l in axh.get_xticklabels():
    lt = l.get_text()
    l.set_text(re.sub('\w+\-', '', lt))
for l in axh.get_yticklabels():
    lt = l.get_text()
    l.set_text(re.sub('.+\-', '', lt))
axh.set_xticklabels(labels=axh.get_xticklabels(), ha='right', rotation=45)
axh.set_yticklabels(labels=axh.get_yticklabels())
for i, t in enumerate(tumors):
    axh.text(i*2+1.0, -1, t, color=lut[t], ha='center')
for i, t in enumerate(g2s):
    axh.text(-2.3, i*2+1.0, t, color=lut_rows[t], va='center')
axh.set_xlabel('')
axh.set_ylabel('')
#pj.easy_save(g.fig, './figures/spearmanr_heatmap_nosc.svg', fmt='svg', dpi=600)
plt.show()

############# Regressoin plots (RU1108 and SC53) ######
vx = 'A2 score'
vy = 'E score (nnPC 1)'
pal = pj.palette_default
for i, df in enumerate(dfs2p[:2]):
    fig, ax = plt.subplots(figsize=(5.3,4))
    fig.subplots_adjust(left=0.17, bottom=0.135, right=0.72)
    sns.regplot(data=df.sample(10), x=vx, y=vy, scatter_kws={'alpha':0.1}, line_kws={'color':'b'}, ax=ax, label='All cells')
    sns.regplot(data=df[df.ASCL1>0].sample(10), x=vx, y=vy, scatter_kws={'alpha':0.1}, line_kws={'color':'r'}, ax=ax, label=r"ASCL1$^+$ cells")
    ax.legend(bbox_to_anchor=(1, 0, 0.2, 0.8))
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(12)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(12)
    ax.xaxis.label.set_size(12)
    ax.xaxis.label.set_color(pal[vx.replace(' score', '')])
    ax.yaxis.label.set_size(12)
    figname = './figures/regplot_'+vx.replace(' ', '_')+'_'+vy.replace(' ', '_')+'_'+tumors[i]
    #pj.easy_save(fig, figname+'.png', fmt='png', dpi=600)
plt.show()



############## Treatment DataFrame for SC53 and SC68 #######
dfs2psc = [df53, df68]
tumors = ['SC53', 'SC68']
g1s = ['E score', 'CDH1 expression']
samp_types = ['Untreated all', 'Untreated ASCL1+', 'Cisplatin all', 'Cisplatin ASCL1+']
#g2s = ['A2 score']
d2psc = []
#for g2, samp_type in product(g2s, samp_types):
for samp_type in samp_types:
    row_data = []
    g2 = 'A2 score'
    for tumor, g1 in product(tumors, g1s):
        df = dfs2psc[tumors.index(tumor)].dropna()
        df = df[df.Treatment == re.sub(' .+', '', samp_type).lower()]
        if 'ASCL1+' in samp_type:
            df = df[df.ASCL1>0]
        g1 = g1.replace(' expression', '').replace('E score', 'E score (nnPC 1)')
        corr, p = stats.spearmanr(df[g1], df[g2])
        if p > 0.05:
            print(g1, g2, tumor, samp_type)
        row_data.append(corr)
    d2psc.append(row_data)
#index = pd.MultiIndex.from_product((g2s, samp_types), names=('Score 2', 'Cell type'))
index = pd.Index(samp_types)
cols = pd.MultiIndex.from_product((tumors, g1s), names=('Tumor', 'Score 1'))
df2psc = pd.DataFrame(d2psc, index=index, columns=cols)
#df2psc.to_csv('treat_corr.csv')

lut = dict(zip(tumors, "bg"))
col_colors = df2psc.columns.get_level_values('Tumor').map(lut)
g = sns.clustermap(df2psc, col_cluster=False, row_cluster=False,
        center=0, cmap='RdBu_r', figsize=(4.2,4.5), linewidth=0.5,
        col_colors=col_colors,# row_colors=row_colors,
        cbar_kws={'label': r'Spearman $\it{r}$', 'orientation': 'horizontal'},
        cbar_pos=(0.7, 0.94, 0.18, 0.03),
        dendrogram_ratio=(0.28, 0.3))
axh = g.ax_heatmap
for l in axh.get_xticklabels():
    lt = l.get_text()
    l.set_text(re.sub('\w+\-', '', lt))
axh.set_xticklabels(labels=axh.get_xticklabels(), ha='right', rotation=45)
axh.set_yticklabels(labels=axh.get_yticklabels())
for i, t in enumerate(tumors):
    axh.text(i*2+1.0, -0.5, t, color=lut[t], ha='center')
axh.set_xlabel('')
axh.set_ylabel('')
#pj.easy_save(g.fig, './figures/spearmanr_treatment_heatmap_nosc.svg', fmt='svg', dpi=600)
plt.show()

########### Treatment Regplots (SC53 and SC68) ########
vx = 'A2 score'
vy = 'E score (nnPC 1)'
for i, df in enumerate(dfs2psc):
    fig, ax = plt.subplots(figsize=(5.3,4))
    fig.subplots_adjust(left=0.15, bottom=0.135, right=0.7)
    sns.regplot(data=df[(df.ASCL1>0)&(df.Treatment=='untreated')], x=vx, y=vy, scatter_kws={'alpha':0.1}, line_kws={'color':'b'}, ax=ax, label='Untreated\n'+r'ASCL1$^+$')
    sns.regplot(data=df[(df.ASCL1>0)&(df.Treatment=='cisplatin')], x=vx, y=vy, scatter_kws={'alpha':0.1}, line_kws={'color':'r'}, ax=ax, label="Cisplatin\ntreated\n"+r"ASCL1$^+$")
    ax.legend(bbox_to_anchor=(1, 0, 0.2, 0.8))
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(12)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(12)
    ax.xaxis.label.set_size(12)
    ax.yaxis.label.set_size(12)
    figname = './figures/regplot_treatment_'+vx.replace(' ', '_')+'_'+vy.replace(' ', '_')+'_'+tumors[i]
    #pj.easy_save(fig, figname+'.png', fmt='png', dpi=600)
plt.show()


