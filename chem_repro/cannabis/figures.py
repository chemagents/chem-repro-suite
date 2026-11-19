"""Scientific views of complete computed tables; no paper values masquerade as calculations."""
import numpy as np
import pandas as pd
from ..common.figures import initialize_style, export_figure


def complete(result, name):
    value=result.answer.tables[name]
    if value.is_preview:
        import hashlib, io
        from ..common.artifacts import artifact_root
        digest=value.full_table_artifact_sha256
        source=next((a for a in result.artifacts if a.sha256==digest and a.media_type=='text/csv'),None)
        if source is None:raise ValueError('Preview lacks its registered complete CSV')
        payload=(artifact_root()/(digest+'.csv')).read_bytes()
        if hashlib.sha256(payload).hexdigest()!=digest:raise ValueError('Full plotting table checksum differs')
        frame=pd.read_csv(io.BytesIO(payload))
        if len(frame)!=value.total_rows or list(frame)!=value.columns:
            raise ValueError('Complete plotting table rows or columns differ')
        return frame
    return pd.DataFrame(value.rows)


def publish(result, fig, name, source, description):
    result.artifacts.extend(export_figure(fig,name,port=7341))
    result.metadata.protocol_parameters.setdefault('generated_figures',[]).append(
        {'name':name,'source':source,'description':description,'evidence_kind':'computed'})
    return result


def docking(result, frame, targets):
    import matplotlib.pyplot as plt
    initialize_style();fig,axes=plt.subplots(2,3,figsize=(6.75,4.5),layout='constrained')
    for ax,(target,(column,_)) in zip(axes.flat,targets.items(),strict=True):
        for value,label,color in [(1,'Pesticides','#0072B2'),(0,'AChE-inactive','#D55E00')]:
            x=np.sort(frame.loc[frame.activity==value,column].to_numpy(float))
            ax.step(x,np.arange(1,len(x)+1)/len(x),where='post',label=f'{label} (n={len(x)})',color=color,lw=1.3)
        ax.set(title=target,xlabel='Docking score (kcal/mol)',ylabel='Cumulative fraction',ylim=(0,1))
    axes[0,0].legend(fontsize=7,frameon=False)
    return publish(result,fig,'cannabis_docking_distributions','pinned labeled source docking scores',
        'Full DS2/DS3 score ECDFs, without a new docking run. Unmapped final DS1 is not substituted.')


def selection(result):
    import matplotlib.pyplot as plt
    spectrum=complete(result,'eigenspectrum');scan=complete(result,'inner_m_scan')
    rank=complete(result,'ranked_residues');metrics={m.name:m.value for m in result.answer.metrics}
    p=result.metadata.protocol_parameters;initialize_style()
    fig,axes=plt.subplots(1,3,figsize=(6.75,2.7),layout='constrained')
    axes[0].plot(spectrum.component,spectrum.eigenvalue,'o',ms=2,color='#0072B2')
    axes[0].axhline(metrics['lambda_plus'],ls='--',color='#D55E00',label='MP upper edge')
    axes[0].set(xlabel='Component',ylabel='Eigenvalue');axes[0].legend(fontsize=6,frameon=False)
    axes[1].plot(scan.m,scan.mean_inner_roc_auc,color='#0072B2')
    axes[1].fill_between(scan.m,scan.mean_inner_roc_auc-scan.std_inner_roc_auc,
                        scan.mean_inner_roc_auc+scan.std_inner_roc_auc,alpha=.18,color='#0072B2')
    axes[1].axvline(metrics['m_star'],ls='--',color='#D55E00')
    axes[1].set(xlabel='Top-m residues',ylabel='Inner ROC-AUC (mean ± SD)')
    gate=rank.loc[rank.in_top_m].sort_values('energy_rank_within_top_m')
    axes[2].plot(gate.energy_rank_within_top_m,gate.cumulative_contribution_fraction,color='#0072B2')
    axes[2].axhline(.98,ls='--',color='#D55E00');axes[2].axvline(metrics['selected_residues'],ls=':',color='black')
    axes[2].set(xlabel='Contribution rank',ylabel='Cumulative contribution',ylim=(0,1.03))
    fig.suptitle(f"{p['target']} · source split {p['split_index']} · training-only selection",fontsize=9)
    return publish(result,fig,f"appendix_a_{p['target'].lower()}_split{p['split_index']:02d}",
        'eigenspectrum, inner_m_scan, ranked_residues',
        'Deterministic training spectrum and 98% gate; inner ROC-AUC band is fold SD, not an external validation interval.')


def fusion(result):
    import matplotlib.pyplot as plt
    frame=complete(result,'fold_summary');initialize_style()
    for scheme,rows in frame.groupby('scheme',sort=True):
        rows=rows.set_index('branch').loc[['structure','rte','product']]
        fig,axes=plt.subplots(1,3,figsize=(6.75,2.8),layout='constrained')
        for ax,key,label in zip(axes,['roc_auc','fpr','recall'],['ROC-AUC','False-positive fraction','Recall'],strict=True):
            ax.errorbar(range(3),rows['mean_'+key],yerr=rows['sd_'+key].fillna(0),fmt='o',capsize=3,color='#0072B2')
            ax.set(xticks=range(3),xticklabels=['DMPNN','RTE','Product'],ylabel=label,ylim=(0,1.03))
        fig.suptitle(f'{scheme} · matched test records · fold mean ± SD',fontsize=9)
        publish(result,fig,'fusion_'+scheme+'_operating_metrics','fold_summary',
                'Three branches on identical held-out records; threshold0.5, fold means and SD; repeated holdouts are not independent OOF.')
    return result


def candidates(result):
    if 'figure4_source_components' not in result.answer.tables:return result
    import matplotlib.pyplot as plt
    frame=complete(result,'figure4_source_components').sort_values('paper_rank');initialize_style()
    fig,ax=plt.subplots(figsize=(6.75,4.4),layout='constrained')
    for delta,col,label,color in [(-.18,'computed_structure_probability','DMPNN','#0072B2'),
                                 (0,'computed_rte_probability','RTE','#D55E00'),
                                 (.18,'computed_product_score','Product','#009E73')]:
        ax.plot(frame[col],np.arange(len(frame))+delta,'o',ms=3,label=label,color=color)
    ax.set(yticks=range(len(frame)),yticklabels=frame.paper_name,xlim=(0,1),xlabel='Reconstructed classification score')
    ax.invert_yaxis();ax.legend(loc='lower right',fontsize=7,frameon=False)
    return publish(result,fig,'mapped_figure4_source_components','figure4_source_components',
        'New model scores for the ten mapped published compounds, ordered by paper rank. Not a recovered final top-ten selection or author component probabilities.')


def toxicity(result):
    if 'cohort_comparisons' not in result.answer.tables:return result
    import matplotlib.pyplot as plt
    frame=complete(result,'cohort_comparisons')
    if frame.empty:return result
    if 'scheme' not in frame:frame=frame.assign(scheme='fixed_public_v1')
    initialize_style()
    for scheme,rows in frame.groupby('scheme',sort=True):
        endpoints=list(rows.endpoint.unique());nrows=int(np.ceil(len(endpoints)/3))
        for field,label,suffix in [('median_prediction','Median prediction','medians'),
                                   ('outside_reference_distance_fraction','Outside-domain fraction','applicability')]:
            fig,axes=plt.subplots(nrows,3,figsize=(6.75,2.2*nrows),layout='constrained',squeeze=False)
            for ax,ep in zip(axes.flat,endpoints):
                g=rows.loc[rows.endpoint==ep].sort_values('cohort')
                short=[str(x).replace('DS1_','').replace('DS2_','').replace('DS3_','').replace('_','\n') for x in g.cohort]
                ax.plot(range(len(g)),g[field],'o',color='#0072B2')
                unit=str(g.unit.iloc[0]) if field=='median_prediction' else 'fraction'
                ax.set(xticks=range(len(g)),xticklabels=short,title=ep.replace('_',' '),ylabel=unit)
                ax.tick_params(axis='x',labelsize=6)
                if suffix=='applicability':ax.set_ylim(0,1)
            for ax in list(axes.flat)[len(endpoints):]:ax.set_visible(False)
            fig.suptitle(f'{scheme} · {label.lower()} · open models',fontsize=9)
            publish(result,fig,f'open_toxicity_{scheme}_{suffix}','cohort_comparisons',
                'Source-cohort point summaries from declared open models; no uncertainty interval or original Syntelly equivalence is implied.')
    return result


def author_replay(result):
    """Every split remains visible; error bars describe SD, not confidence intervals."""
    import matplotlib.pyplot as plt
    per=complete(result,'computed_per_split')
    methods=['dmpnn','catboost_rmt95_standalone','svm_rmt95_standalone','product_rmt95_veto']
    labels=['Source\nblend','CatBoost\nRMT95','SVM\nRMT95','Product']
    initialize_style()
    fig,axes=plt.subplots(1,3,figsize=(6.75,2.8),layout='constrained')
    for letter,ax,key,ylabel,scale in zip('abc',axes,['roc_auc','fpr','recall'],
            ['ROC-AUC','False-positive rate (%)','Recall (%)'],[1,100,100],strict=True):
        for i,method in enumerate(methods):
            values=per.loc[per.method==method,key].to_numpy(float)*scale
            ax.scatter(i+np.linspace(-.12,.12,len(values)),values,s=4,c='#9EA3A8',alpha=.6,zorder=1)
            ax.errorbar(i,values.mean(),yerr=values.std(ddof=1),fmt='o',ms=4,c='#0072B2',capsize=3,zorder=2)
        ax.set(xticks=range(4),xticklabels=labels,ylabel=ylabel)
        ax.tick_params(axis='x',labelsize=7)
        ax.text(-.13,1.03,f'({letter})',transform=ax.transAxes,weight='bold',fontsize=9)
        if key!='roc_auc':ax.set_ylim(0,100)
    fig.suptitle('Deposited author code · 30 repeated holdouts · mean ± SD',fontsize=9)
    return publish(result,fig,'cannabis_author_rmt95_validation','computed_per_split',
        'Original-code refits: Source blend means validation-selected DMPNN/HGB. Each grey point is one source holdout; blue whiskers show sample SD across30splits. '
        'RMT95 product trades lower FPR for lower recall at threshold0.5. Source global-label selection is preserved; '
        'this is not independent validation or the final RTE390 geometric mixture.')
