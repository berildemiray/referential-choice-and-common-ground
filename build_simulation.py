"""Python mirror to produce the synthetic dataset and verify descriptive results.
An R analysis workflow is included separately. This file documents the executed simulation.
"""
from pathlib import Path
import numpy as np, pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.special import expit

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data'; RESULTS=ROOT/'results'
OUT.mkdir(exist_ok=True); RESULTS.mkdir(exist_ok=True)
RNG=np.random.default_rng(20261019)
N_SPEAKERS=48; N_ITEMS=12
NOISE=.15; COST=.50; ALPHA=1.8
u_part=RNG.normal(0,.50,size=N_SPEAKERS)
u_item=RNG.normal(0,.25,size=N_ITEMS)
rows=[]
for participant in range(1,N_SPEAKERS+1):
    for item in range(1,N_ITEMS+1):
        for shared in (0,1):
            for surprise in (0,1):
                # In the 'not shared' condition listener uses a neutral prior.
                prior=.5 if shared==0 else (.1 if surprise==1 else .9)
                # Reduced form carries no referent-identity evidence; full NP is noisy but diagnostic.
                listener_reduced=prior
                listener_explicit=prior*(1-NOISE)/(prior*(1-NOISE)+(1-prior)*NOISE)
                delta=ALPHA*(np.log(listener_explicit)-np.log(listener_reduced)-COST)+u_part[participant-1]+u_item[item-1]
                p=expit(delta)
                explicit=int(RNG.random() < p)
                rows.append([f'P{participant:02d}',f'I{item:02d}', shared, surprise,
                             'shared' if shared else 'not_shared',
                             'surprising' if surprise else 'predictable',
                             prior,listener_reduced,listener_explicit,p,explicit])
cols=['participant','item','shared','surprise','common_ground','event_type','prior_target','L0_reduced','L0_explicit','speaker_p_explicit','explicit_choice']
df=pd.DataFrame(rows,columns=cols)
df.to_csv(OUT/'synthetic_reference_trials.csv',index=False)

smry=(df.groupby(['common_ground','event_type'],sort=False)['explicit_choice']
        .agg(n='size',full_np='sum',rate='mean').reset_index())
smry.to_csv(RESULTS/'condition_summary.csv',index=False)

# Standard inference: participant and item fixed effects, with participant clustered SE.
fit=smf.glm('explicit_choice ~ surprise * shared + C(participant) + C(item)',
            data=df, family=sm.families.Binomial()).fit(cov_type='cluster',cov_kwds={'groups':df['participant']})
terms=['surprise','shared','surprise:shared']
model=pd.DataFrame({'term':terms,
 'log_odds':[fit.params[t] for t in terms],
 'SE_cluster_participant':[fit.bse[t] for t in terms],
 'p_cluster_participant':[fit.pvalues[t] for t in terms],
 'odds_ratio':[np.exp(fit.params[t]) for t in terms]})
model.to_csv(RESULTS/'model_coefficients.csv',index=False)

piv=df.groupby(['participant','shared','surprise'])['explicit_choice'].mean().unstack(['shared','surprise'])
# (shared=1 surprising - predictable) - (shared=0 surprising - predictable)
did = (piv[(1,1)]-piv[(1,0)]-piv[(0,1)]+piv[(0,0)]).mean()
shared_effect=(piv[(1,1)]-piv[(1,0)]).mean()
notshared_effect=(piv[(0,1)]-piv[(0,0)]).mean()
B=3000
rng=np.random.default_rng(71173)
bootstrap=np.empty((B,3))
array=np.asarray(piv[[ (1,1),(1,0),(0,1),(0,0) ]]); n=len(array)
for b in range(B):
    a=array[rng.integers(0,n,size=n)]
    bootstrap[b]=[(a[:,0]-a[:,1]).mean(),(a[:,2]-a[:,3]).mean(),(a[:,0]-a[:,1]-a[:,2]+a[:,3]).mean()]
ci=np.quantile(bootstrap,[.025,.975],axis=0)
contrast=pd.DataFrame({'contrast':['Surprise effect | shared knowledge', 'Surprise effect | not shared', 'Difference-in-differences'],
'estimate_pp':[shared_effect*100,notshared_effect*100,did*100],
'CI_low_pp':ci[0,:]*100,'CI_high_pp':ci[1,:]*100})
contrast.to_csv(RESULTS/'cluster_bootstrap_contrasts.csv',index=False)

# Noisy-channel sensitivity: analytical P(full NP) without random effects.
sens=[]
for noise in [0.05,0.10,0.15,0.20,0.30,0.40]:
 for shared in [0,1]:
  for surprise in [0,1]:
   prior=.5 if not shared else (.1 if surprise else .9)
   full=prior*(1-noise)/(prior*(1-noise)+(1-prior)*noise)
   prob=expit(ALPHA*(np.log(full)-np.log(prior)-COST))
   sens.append([noise,shared,surprise,prob])
pd.DataFrame(sens,columns=['noise','shared','surprise','expected_full_np_rate']).to_csv(RESULTS/'noise_sensitivity.csv',index=False)

print('N=',len(df));print(smry.to_string(index=False))
print('\n' + model.to_string(index=False,float_format=lambda x: f'{x:.4f}'))
print('\n' + contrast.to_string(index=False,float_format=lambda x: f'{x:.2f}'))
