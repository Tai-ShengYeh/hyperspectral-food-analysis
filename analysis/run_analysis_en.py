"""English-labeled figures for the EN hyperspectral video. Same real analysis, English labels.
Outputs PNGs -> ../video_en/assets/images/  (numbers identical to results.json)
"""
import sys, codecs, json
if sys.platform.startswith('win'):
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach()); sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, r2_score, mean_squared_error
from sklearn.cross_decomposition import PLSRegression
from pathlib import Path

ROOT = Path(__file__).resolve().parent
IMG = ROOT.parent / "video_en" / "assets" / "images"; IMG.mkdir(parents=True, exist_ok=True)
DATA = ROOT / "data" / "SpectroFood_dataset.csv"
TEAL="#0E7C7B"; CORAL="#E36414"; GOLD="#C8941F"; PURPLE="#7c3aed"; INK="#1A1A1A"; GRID="#dddddd"
FOOD_COLOR={"Apple":CORAL,"Broccoli":TEAL,"Leek":GOLD,"Mushroom":PURPLE}
PREFIX={"A":"Apple","B":"Broccoli","L":"Leek","M":"Mushroom"}; ORDER=["Apple","Broccoli","Leek","Mushroom"]
plt.rcParams.update({"figure.facecolor":"white","axes.facecolor":"white","savefig.facecolor":"white",
    "axes.edgecolor":"#999","font.size":13,"axes.titlesize":15,"font.family":"DejaVu Sans",
    "axes.grid":True,"grid.color":GRID,"grid.linewidth":0.6,"axes.unicode_minus":False})

df = pd.read_csv(DATA)
ids = df.iloc[:,0].astype(str); pref = ids.str.extract(r'^([ABLM])\d')[0]
wl = np.array([float(c) for c in df.columns[2:]])
Xall = df.iloc[:,2:].apply(pd.to_numeric, errors='coerce').values.astype(float)
dm_all = pd.to_numeric(df.iloc[:,1], errors='coerce').values
common = np.ones(Xall.shape[1], bool)
for p in "ABLM": common &= ~np.isnan(Xall[(pref==p).values]).any(axis=0)
wlc = wl[common]
rows = pref.isin(list("ABLM")).values
food = pref[rows].map(PREFIX).values; Xc = Xall[rows][:, common]; dm = dm_all[rows]
ok = ~np.isnan(Xc).any(axis=1); Xc, food, dm = Xc[ok], food[ok], dm[ok]
def snv(a): return (a-a.mean(1,keepdims=True))/a.std(1,keepdims=True)
Xsnv = snv(Xc)

# FIG1
fig,axes=plt.subplots(1,2,figsize=(12,4.6))
for f in ORDER:
    m=food==f; mu=Xc[m].mean(0); sd=Xc[m].std(0)
    axes[0].plot(wlc,mu,color=FOOD_COLOR[f],lw=2,label=f); axes[0].fill_between(wlc,mu-sd,mu+sd,color=FOOD_COLOR[f],alpha=0.12)
axes[0].axvspan(398,700,color="#fde047",alpha=0.10); axes[0].axvspan(700,773,color="#fca5a5",alpha=0.12)
axes[0].set_title("Raw Reflectance Spectra (mean ± SD)",weight="bold")
axes[0].set_xlabel("Wavelength (nm)"); axes[0].set_ylabel("Reflectance"); axes[0].legend(fontsize=11,loc="upper left")
axes[0].text(545,axes[0].get_ylim()[1]*0.05,"Visible",fontsize=10,color="#a16207",ha="center")
axes[0].text(737,axes[0].get_ylim()[1]*0.05,"Red edge",fontsize=9,color="#b91c1c",ha="center")
for f in ORDER:
    m=food==f; axes[1].plot(wlc,Xsnv[m].mean(0),color=FOOD_COLOR[f],lw=2,label=f)
axes[1].set_title("After SNV Normalization (removes scatter)",weight="bold")
axes[1].set_xlabel("Wavelength (nm)"); axes[1].set_ylabel("SNV Reflectance"); axes[1].legend(fontsize=11)
plt.tight_layout(); plt.savefig(IMG/"fig1_spectra.png",dpi=150); plt.close(); print("[OK] fig1")

# FIG2
pca=PCA(n_components=10).fit(Xsnv); sc=pca.transform(Xsnv); ev=pca.explained_variance_ratio_
fig,axes=plt.subplots(1,2,figsize=(12,4.6))
axes[0].bar(range(1,11),ev*100,color=TEAL,alpha=0.8); axes[0].plot(range(1,11),np.cumsum(ev)*100,color=CORAL,marker="o",lw=2)
axes[0].set_title("PCA Explained Variance",weight="bold"); axes[0].set_xlabel("Principal Component"); axes[0].set_ylabel("Variance (%)")
axes[0].text(3.2,np.cumsum(ev)[1]*100-8,f"First 2 PCs = {ev[:2].sum()*100:.1f}%",fontsize=11,color=CORAL)
for f in ORDER:
    m=food==f; axes[1].scatter(sc[m,0],sc[m,1],s=16,color=FOOD_COLOR[f],alpha=0.6,label=f,edgecolors="none")
axes[1].set_title("PCA Scatter: Four Foods Cluster Naturally",weight="bold")
axes[1].set_xlabel(f"PC1 ({ev[0]*100:.1f}%)"); axes[1].set_ylabel(f"PC2 ({ev[1]*100:.1f}%)"); axes[1].legend(fontsize=11)
plt.tight_layout(); plt.savefig(IMG/"fig2_pca.png",dpi=150); plt.close(); print("[OK] fig2")

# FIG3
Xtr,Xte,ytr,yte=train_test_split(Xsnv,food,test_size=0.25,random_state=42,stratify=food)
svm=make_pipeline(StandardScaler(),SVC(kernel="rbf",C=10,gamma="scale")).fit(Xtr,ytr); pred=svm.predict(Xte)
acc=accuracy_score(yte,pred); cm=confusion_matrix(yte,pred,labels=ORDER)
fig,ax=plt.subplots(figsize=(6.2,5.6)); ax.imshow(cm,cmap="BuGn")
ax.set_xticks(range(4)); ax.set_yticks(range(4)); ax.set_xticklabels(ORDER,rotation=15); ax.set_yticklabels(ORDER)
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
ax.set_title(f"SVM Confusion Matrix (test n={len(yte)})\nAccuracy = {acc*100:.1f}%",weight="bold")
for i in range(4):
    for j in range(4): ax.text(j,i,cm[i,j],ha="center",va="center",color="white" if cm[i,j]>cm.max()/2 else INK,fontsize=14,weight="bold")
plt.tight_layout(); plt.savefig(IMG/"fig3_confusion.png",dpi=150); plt.close(); print(f"[OK] fig3 acc={acc*100:.1f}")

# FIG4 (leek full NIR)
mL=(pref=="L").values; XL=Xall[mL]; dmL=dm_all[mL]; validL=~np.isnan(XL).any(axis=0)
XLv=XL[:,validL]; wlL=wl[validL]; okL=(~np.isnan(XLv).any(axis=1))&(~np.isnan(dmL)); XLv,dmL=XLv[okL],dmL[okL]*100
XLs=snv(XLv); nc=12; yp=cross_val_predict(PLSRegression(n_components=nc),XLs,dmL,cv=10).ravel()
r2=r2_score(dmL,yp); rmse=np.sqrt(mean_squared_error(dmL,yp))
fig,ax=plt.subplots(figsize=(6.0,5.6)); ax.scatter(dmL,yp,s=20,color=GOLD,alpha=0.6,edgecolors="none",label="Leek samples")
lim=[min(dmL.min(),yp.min())-1,max(dmL.max(),yp.max())+1]; ax.plot(lim,lim,"--",color=INK,lw=1.2,alpha=0.7,label="Ideal 1:1")
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_title(f"PLS Regression: Dry Matter (Leek, {validL.sum()} bands to {wlL.max():.0f}nm)\n10-fold CV: R²={r2:.3f}, RMSE={rmse:.2f}%",weight="bold")
ax.set_xlabel("Actual Dry Matter (%)"); ax.set_ylabel("Predicted Dry Matter (%)"); ax.legend(fontsize=11)
plt.tight_layout(); plt.savefig(IMG/"fig4_regression.png",dpi=150); plt.close(); print(f"[OK] fig4 R2={r2:.3f} RMSE={rmse:.2f}")
print("EN figures done.")
