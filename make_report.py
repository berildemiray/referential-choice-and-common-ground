from pathlib import Path
import pandas as pd, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, KeepTogether, PageBreak, HRFlowable
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

ROOT=Path(__file__).resolve().parents[1]
fig=ROOT/'figures'; fig.mkdir(exist_ok=True)
res=ROOT/'results';df=pd.read_csv(ROOT/'data/synthetic_reference_trials.csv')
summ=pd.read_csv(res/'condition_summary.csv'); sens=pd.read_csv(res/'noise_sensitivity.csv')
contrast=pd.read_csv(res/'cluster_bootstrap_contrasts.csv'); model=pd.read_csv(res/'model_coefficients.csv')
DARK='#123146'; TEAL='#176B7B'; GOLD='#D59B40'; PALE='#E9F2F2'; GREY='#586C74'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.labelcolor':DARK,'text.color':DARK,'xtick.color':GREY,'ytick.color':GREY})

# Figure 1: 2x2 condition means
fig1,ax=plt.subplots(figsize=(8.15,3.7),dpi=170)
order=['not_shared','shared']; x=np.arange(2); w=.31
for offset,event,color in [(-w/2,'predictable',TEAL),(w/2,'surprising',GOLD)]:
    vals=[float(summ[(summ.common_ground==g)&(summ.event_type==event)].rate.iloc[0]) for g in order]
    bars=ax.bar(x+offset,vals,width=w,color=color,label=event.capitalize(),edgecolor='white',linewidth=.7)
    for bar,val in zip(bars,vals): ax.text(bar.get_x()+bar.get_width()/2,val+.022,f'{100*val:.1f}%',ha='center',fontsize=10,fontweight='bold')
ax.set_xticks(x,['Not shared','Shared']);ax.set_ylabel('Explicit NP selection (%)');ax.yaxis.set_major_formatter(PercentFormatter(1));ax.set_ylim(0,.88)
ax.set_title('Model-generated expression choices',fontweight='bold',fontsize=13,loc='left',pad=14)
ax.legend(frameon=False,ncol=2,loc='upper left');ax.grid(axis='y',alpha=.16);ax.set_axisbelow(True)
fig1.tight_layout();fig1.savefig(fig/'condition_means.png',bbox_inches='tight',facecolor='white');plt.close(fig1)

# Figure 2: sensitivity to noisy word identification
fig2,ax=plt.subplots(figsize=(8.15,3.6),dpi=170)
for (shared,surprise),lab,c,lsty in [((1,1),'Shared | surprising',GOLD,'-'),((1,0),'Shared | predictable',TEAL,'-'),((0,1),'Not shared | both',GREY,'--')]:
    zz=sens[(sens.shared==shared)&(sens.surprise==surprise)]
    ax.plot(zz.noise,zz.expected_full_np_rate,lsty,color=c,lw=2.6,label=lab,marker='o',markersize=5)
ax.set_title('What changes when the channel becomes less reliable?',fontweight='bold',fontsize=13,loc='left',pad=14)
ax.set_xlabel('Probability of identification noise');ax.set_ylabel('Analytical P(explicit NP)');ax.set_ylim(.15,1.02)
ax.yaxis.set_major_formatter(PercentFormatter(1));ax.xaxis.set_major_formatter(PercentFormatter(1))
ax.legend(frameon=False,loc='upper right',fontsize=9);ax.grid(alpha=.17);ax.set_axisbelow(True)
fig2.tight_layout();fig2.savefig(fig/'channel_sensitivity.png',bbox_inches='tight',facecolor='white');plt.close(fig2)

# Figure 3: participant-block bootstrap with CIs
fig3,ax=plt.subplots(figsize=(7.8,3.0),dpi=170)
labels=['Shared-knowledge surprise effect','No-shared-knowledge surprise effect','Difference-in-differences']
y=np.array([2,1,0]);mid=contrast.estimate_pp.to_numpy();lo=contrast.CI_low_pp.to_numpy();hi=contrast.CI_high_pp.to_numpy()
for ii in range(3):
  ax.plot([lo[ii],hi[ii]],[y[ii],y[ii]],color=TEAL,lw=3)
  ax.plot(mid[ii],y[ii],'o',color=GOLD if ii==2 else TEAL,markersize=9)
  ax.text(hi[ii]+1,y[ii],f'{mid[ii]:+.1f} pp',va='center',fontsize=9)
ax.set_yticks(y,labels);ax.set_xlim(-12,68);ax.axvline(0,color=GREY,lw=.8,alpha=.5)
ax.set_xlabel('Percentage-point difference (95% participant-block bootstrap CI)')
ax.grid(axis='x',alpha=.12);ax.set_axisbelow(True);fig3.tight_layout()
fig3.savefig(fig/'effect_intervals.png',bbox_inches='tight',facecolor='white');plt.close(fig3)

# PDF
for nm,path in [('Lato','/usr/share/fonts/truetype/lato/Lato-Regular.ttf'),('LatoBold','/usr/share/fonts/truetype/lato/Lato-Bold.ttf'),('LatoItalic','/usr/share/fonts/truetype/lato/Lato-Italic.ttf')]:pdfmetrics.registerFont(TTFont(nm,path))
pdfmetrics.registerFontFamily('Lato', normal='Lato', bold='LatoBold', italic='LatoItalic', boldItalic='LatoBold')
C=lambda s:colors.HexColor(s)
styles={
 'title':ParagraphStyle('T',fontName='LatoBold',fontSize=21,leading=25,textColor=C(DARK),spaceAfter=12),
 'subtitle':ParagraphStyle('ST',fontName='Lato',fontSize=12,leading=17,textColor=C(TEAL),spaceAfter=13),
 'eyebrow':ParagraphStyle('EY',fontName='LatoBold',fontSize=9,leading=12,textColor=C(TEAL),spaceAfter=14),
 'h1':ParagraphStyle('H1',fontName='LatoBold',fontSize=13,leading=17,textColor=C(DARK),spaceBefore=15,spaceAfter=7,keepWithNext=1),
 'h2':ParagraphStyle('H2',fontName='LatoBold',fontSize=10.6,leading=14,textColor=C(TEAL),spaceBefore=9,spaceAfter=4),
 'body':ParagraphStyle('B',fontName='Lato',fontSize=9.35,leading=14.7,textColor=C(DARK),spaceAfter=8),
 'small':ParagraphStyle('S',fontName='Lato',fontSize=8.4,leading=12.1,textColor=C(GREY),spaceAfter=5),
 'caption':ParagraphStyle('CAP',fontName='LatoItalic',fontSize=8.25,leading=11.5,textColor=C(GREY),spaceAfter=11),
 'quote':ParagraphStyle('Q',fontName='LatoBold',fontSize=10.5,leading=16,textColor=C(DARK),leftIndent=13,rightIndent=12,spaceBefore=8,spaceAfter=12,borderColor=C(TEAL),borderWidth=2,borderPadding=11),
 'table':ParagraphStyle('CT',fontName='Lato',fontSize=8.5,leading=11,textColor=C(DARK)),
 'tablehead':ParagraphStyle('HT',fontName='LatoBold',fontSize=8.2,leading=11,textColor=colors.white),
}
S=[]
def p(t,st='body'):S.append(Paragraph(t,styles[st]))
def h(t):p(t,'h1')
def ttable(rows,widths=None):
    data=[]
    for i,r in enumerate(rows): data.append([Paragraph(str(cell),styles['tablehead' if i==0 else 'table']) for cell in r])
    tab=Table(data,colWidths=widths,hAlign='LEFT',repeatRows=1)
    tab.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),C(DARK)),('VALIGN',(0,0),(-1,-1),'TOP'),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,C('#F0F6F5')]),('BOTTOMPADDING',(0,0),(-1,-1),8),('TOPPADDING',(0,0),(-1,-1),8),('LEFTPADDING',(0,0),(-1,-1),9),('RIGHTPADDING',(0,0),(-1,-1),8),('LINEBELOW',(0,-1),(-1,-1),.55,C('#D5E5E5'))]))
    S.append(tab);S.append(Spacer(1,9))
def im(name,width=475):
    from PIL import Image as PIL
    path=fig/name; w,h=PIL.open(path).size
    S.append(Image(str(path),width=width,height=width*h/w,hAlign='CENTER'))

p('COMPUTATIONAL PRAGMATICS  /  RESEARCH NOTE  /  OCTOBER 2026','eyebrow')
p('When are pronouns too risky?','title')
p('A simulation-based study of event predictability, common ground, and reference choice','subtitle')
p('<b>Study type:</b> computational simulation with a 2 × 2 factorial design and synthetic trial-level observations. <b>No human participants were recruited.</b>','small')
S.append(HRFlowable(width='100%',thickness=1,color=C('#B8D2D1'),spaceBefore=9,spaceAfter=13))
h('Abstract')
p('Pronouns are economical but can be ambiguous. This simulation explores whether event predictability and shared knowledge jointly influence a model speaker’s choice between explicit and reduced reference. A 2 × 2 design generates 2,304 synthetic choices from 48 virtual agents across 12 items. With shared knowledge, explicit reference is selected in <b>74.8%</b> of surprising trials versus <b>30.7%</b> of predictable trials; without shared knowledge, rates are <b>50.5%</b> and <b>50.2%</b>. A participant-block bootstrap estimates an interaction of <b>+43.8 percentage points</b> (95% CI [35.2, 51.9]). The results illustrate a specified computational mechanism, not human speech behaviour.')
h('1  |  Motivation and conceptual context')
p('Work on reference production connects pronoun choice to discourse accessibility and expectations (Arnold, 2001; Achimova et al., 2022). Shared knowledge can help speakers adapt descriptions to their interlocutors (Clark &amp; Brennan, 1991), and probabilistic pragmatic models formalize trade-offs between informativeness and expression cost (Frank &amp; Goodman, 2012). This simulation isolates how event predictability interacts with knowledge alignment. It is <b>not</b> a replication or reanalysis of published human data.')
p('<b>Question.</b> In a noisy channel, does predictability matter more for referential expression choice when a listener shares the speaker’s expectations?','quote')
h('2  |  Hypotheses')
p('<b>H1.</b> With shared expectations, surprising events should increase explicit noun-phrase (NP) choice compared with predictable events. <b>H2.</b> Without shared expectations, the effect of speaker-side event surprise should disappear because the listener has no corresponding expectation. <b>H3.</b> Increasing identification noise should diminish the advantage of an explicit description in this simplified model.')
h('3  |  Design')
p('Each of 48 simulated agents completed one trial for each of 12 items in every factorial cell (48 × 12 × 4 = 2,304 trials). The item and agent effects were drawn once, then carried across cells, preserving the repeated-measures structure. The binary outcome codes explicit NPs as 1 and reduced forms as 0.','small')
ttable([['Manipulation','Level A','Level B'],['Event status','Predictable','Surprising'],['Knowledge alignment','Shared: listener knows learned event probabilities','Not shared: listener assigns equal prior probability to alternatives']],[124,181,181])

S.append(Spacer(1,15))
p('RESEARCH DESIGN  /  MODEL SPECIFICATION','eyebrow')
h('4  |  Generative assumptions')
p('Each scenario contains two possible fleeing referents. A reduced expression (e.g., “it fled”) does not identify which referent fled; an explicit expression (e.g., “the blue cell fled”) adds an identifying signal. A listener starts with probability p that the intended referent is correct: p = 0.90 for predictable shared events, p = 0.10 for surprising shared events, and p = 0.50 when expectations are not shared.')
p('We assume that the identifying signal can be corrupted with probability ε = 0.15. The resulting listener success probabilities, given the correct target, are:')
p('<b>Reduced form:</b> L(referent | reduced) = p<br/><b>Explicit form:</b> L(referent | explicit) = p(1 − ε) / [p(1 − ε) + (1 − p) ε]')
p('The model selects an explicit phrase according to a logistic choice rule based on relative log listener success and an explicitness cost:')
p('<b>P(explicit)</b> = logistic{ α [log L(explicit) − log L(reduced) − c] + b_agent + b_item }, with α = 1.8, c = 0.50, agent effects ~ N(0, 0.50²), and item effects ~ N(0, 0.25²).')
p('Parameters are fixed for an illustrative demonstration rather than estimated from human production data. In particular, the model encodes the informational advantage of overt reference by assumption; simulated significance is therefore not an empirical discovery.','small')
h('5  |  Analysis strategy')
p('The analysis reports cell means, an adjusted logistic model with event status, knowledge alignment, their interaction and agent/item fixed effects, and a participant-block bootstrap (3,000 resamples) for raw-percentage-point contrasts. The executed numerical analysis used Python; a corresponding R script supplied with the project includes base-R statistical analyses and optional participant-cluster robust SEs and mixed-effects checks. The R code could not be executed in the available environment, which is disclosed in the README.')
h('6  |  Descriptive findings')
im('condition_means.png',width=480)
p('Figure 1. Observed proportions in the <b>synthetic model-generated dataset</b>. Each bar has 576 trials. Differences are simulated speaker choices, not experimental measurements from people.','caption')
ttable([['Knowledge','Predictable','Surprising','Surprise minus predictable'],['Not shared','50.2% (289/576)','50.5% (291/576)','+0.3 pp'],['Shared','30.7% (177/576)','74.8% (431/576)','+44.1 pp']],[99,115,115,157])
S.append(Spacer(1,15))
p('FINDINGS  /  MODEL VALIDATION','eyebrow')
h('7  |  Interaction and interval estimates')
im('effect_intervals.png',width=483)
p('Figure 2. Participant-block bootstrap 95% intervals from the synthetic dataset (3,000 draws, resampling whole virtual speakers).','caption')
p('For shared knowledge, the surprise effect was <b>+44.1 pp</b>, 95% CI [39.1, 49.1]. Without shared knowledge it was <b>+0.3 pp</b>, 95% CI [−5.0, 5.7]. The difference-in-differences was <b>+43.8 pp</b>, 95% CI [35.2, 51.9].')
p('An adjusted logistic regression with agent and item fixed effects and agent-clustered standard errors yielded a surprise × shared-knowledge interaction of <b>2.05 log-odds units</b> (cluster SE 0.21; odds ratio 7.74; p &lt; .001). This p value only captures simulated variation around the pre-defined mechanism. It should <b>not</b> be interpreted as independent empirical support for a psycholinguistic theory.')
h('8  |  Sensitivity to noise')
im('channel_sensitivity.png',width=483)
p('Figure 3. Analytical reference-choice predictions as channel noise changes while cost and rationality remain fixed; curves omit agent/item heterogeneity to isolate the mechanism.','caption')
p('The predicted probability of explicit reference for shared surprising events drops from 92.7% at 5% identification noise to 43.6% at 40% noise. The no-shared-knowledge curves overlap exactly by construction; their priors are identical for predictable and surprising states.')
h('9  |  What the results mean')
p('The simulation makes a specific, falsifiable <i>computational prediction</i>: if listener expectations are uninformative, speaker-side surprise alone need not favor explicit reference. However, that statement describes the model; actual people may track partner knowledge imperfectly, exploit other pragmatic cues, and use more than two expression types. This result is best treated as a computational proof of concept that motivates future empirical testing, not as evidence of how humans actually speak.')
S.append(Spacer(1,15))
p('LIMITATIONS  /  REPRODUCIBILITY','eyebrow')
h('10  |  Limitations and methodological reflection')
p('<b>Synthetic evidence.</b> All observations were generated from assumptions designed into the model. A significant interaction confirms the simulator is behaving as specified; it does not establish a new human cognitive effect. <b>Restricted language.</b> Only two response classes are represented (explicit vs reduced), whereas real productions include pronouns, zero anaphora, names, and many syntactic alternatives.')
p('<b>World knowledge.</b> The shared condition assigns the correct prior to the listener; the unshared condition uses a fixed 50/50 prior. Real partner knowledge is dynamic and may be inferred over time. <b>Transmission assumptions.</b> Errors are symmetric and referent identity is treated as a single noisy signal; actual auditory ambiguity, phonological repair, prosody and gesture are much richer. <b>Model estimation.</b> The parameters were chosen for interpretability rather than fitted to observed responses, so sensitivity results should not be taken as calibrated forecasts.')
p('<b>Inference.</b> Logistic regression includes item and participant fixed effects; participant-block bootstrap respects the repeated-measures design. In an actual psycholinguistic experiment, confirmatory inference would require planned sampling, ethical approval, preregistered hypotheses, an appropriate random-effects structure and evidence of stimulus validity.')
h('11  |  Reproducible computational workflow')
ttable([['Stage','What is delivered'],['Experimental design','Documented 2 × 2 within-agent factorial manipulation; transparent assumptions and stimulus-item repetition.'],['Data generation','2,304 synthetic trial-level records with agent ID, item ID, priors, listener probabilities, theoretical probabilities, and sampled choices.'],['Statistical analysis','Cell means; logistic regression with agent and item fixed effects; participant-level bootstrap CIs; channel-noise sensitivity.'],['R implementation','Self-contained R analysis script for the included CSV, with optional package-based cluster robust SE and GLMM check.'],['Auditing','Executed Python mirror and CSV outputs. R execution is <b>not</b> claimed.']],[145,341])
h('12  |  Extensions and open questions')
p('A useful next step would be to compare model predictions with observed referring expressions from speakers in controlled discourse settings. The simple neutral prior assigned to unshared expectations could be replaced with a learned or uncertain listener model, and expression choice could be expanded beyond a binary explicit/reduced contrast. Fitting the parameters to behavioural data, instead of choosing them for illustration, would help distinguish assumptions that are computationally convenient from patterns that generalize to language use.')
h('References')
p('Achimova, A., Stegemann-Philipps, C., Winkler, S., &amp; Butz, M. V. (2022). <i>Anaphoric reference in descriptions of surprising events.</i> Proceedings of Sinn und Bedeutung 26, 20–34. https://asya-achimova.github.io/files/2022_Achimova_et_al_Anaphoric_reference.pdf','small')
p('Arnold, J. E. (2001). The effect of thematic roles on pronoun use and frequency of reference continuation. <i>Discourse Processes, 31</i>(2), 137–162. DOI: 10.1207/S15326950DP3102_02','small')
p('Clark, H. H., &amp; Brennan, S. E. (1991). Grounding in communication. In <i>Perspectives on Socially Shared Cognition</i>. DOI: 10.1037/10096-006','small')
p('Frank, M. C., &amp; Goodman, N. D. (2012). Predicting pragmatic reasoning in language games. <i>Science, 336</i>(6084), 998. DOI: 10.1126/science.1218633','small')
p('Companion code, synthetic data, model estimates, and figures are included in the project files. The initial package was developed with AI assistance. Numerical results derive from the executed Python analysis; the companion R script is supplied but has not been validated by execution.','small')

pdf=ROOT/'Independent_Research_Report.pdf'
def footer(canvas,doc):
    canvas.saveState();w,h=A4
    canvas.setStrokeColor(C('#CFDFDE'));canvas.line(53,45,w-53,45)
    canvas.setFont('Lato',8);canvas.setFillColor(C(GREY))
    canvas.drawString(53,32,'COMPUTATIONAL MODEL  •  SYNTHETIC DATA')
    canvas.drawRightString(w-53,32,f'{doc.page}')
    canvas.restoreState()
SimpleDocTemplate(str(pdf),pagesize=A4,rightMargin=53,leftMargin=53,topMargin=48,bottomMargin=55,title='When are pronouns too risky? – Computational Pragmatics',author='Computational Pragmatics Study').build(S,onFirstPage=footer,onLaterPages=footer)
print(pdf)
