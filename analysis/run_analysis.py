"""
SpectroFood hyperspectral analysis — REAL figures + numbers for the teaching video.
Dataset: Zenodo 8362947 (apple/broccoli/leek/mushroom Vis-NIR spectra + dry matter).
The 4 foods were measured with different cameras; we use the 141 COMMON Vis bands
(398-773 nm) for cross-food comparison/classification, and full leek spectra for the
NIR dry-matter regression demo.
Outputs: PNG figures -> ../video_zh/assets/images/ , results.json
"""
import sys, codecs, json
if sys.platform.startswith('win'):
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())

import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
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
IMG = ROOT.parent / "video_zh" / "assets" / "images"
IMG.mkdir(parents=True, exist_ok=True)
DATA = ROOT / "data" / "SpectroFood_dataset.csv"

# register GenSeki (CJK) font for matplotlib so Chinese labels render
FONT = ROOT.parent / "video_zh" / "assets" / "fonts" / "GenSekiGothic2TW-M.otf"
if FONT.exists():
    font_manager.fontManager.addfont(str(FONT))
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(FONT)).get_name()
plt.rcParams["axes.unicode_minus"] = False

TEAL="#0E7C7B"; CORAL="#E36414"; GOLD="#C8941F"; PURPLE="#7c3aed"; INK="#1A1A1A"; GRID="#dddddd"
FOOD_COLOR={"Apple":CORAL,"Broccoli":TEAL,"Leek":GOLD,"Mushroom":PURPLE}
FOOD_ZH={"Apple":"蘋果","Broccoli":"花椰菜","Leek":"韭蔥","Mushroom":"蘑菇"}
PREFIX={"A":"Apple","B":"Broccoli","L":"Leek","M":"Mushroom"}
ORDER=["Apple","Broccoli","Leek","Mushroom"]
plt.rcParams.update({
    "figure.facecolor":"white","axes.facecolor":"white","savefig.facecolor":"white",
    "axes.edgecolor":"#999","font.size":13,"axes.titlesize":15,
    "axes.grid":True,"grid.color":GRID,"grid.linewidth":0.6,
})

# ---------------- Load ----------------
df = pd.read_csv(DATA)
ids = df.iloc[:,0].astype(str)
pref = ids.str.extract(r'^([ABLM])\d')[0]
wl = np.array([float(c) for c in df.columns[2:]])
Xall = df.iloc[:,2:].apply(pd.to_numeric, errors='coerce').values.astype(float)
dm_all = pd.to_numeric(df.iloc[:,1], errors='coerce').values

# common valid bands across all 4 foods
common = np.ones(Xall.shape[1], bool)
for p in "ABLM":
    m=(pref==p).values
    common &= ~np.isnan(Xall[m]).any(axis=0)
wlc = wl[common]

rows = pref.isin(list("ABLM")).values
food = pref[rows].map(PREFIX).values
Xc = Xall[rows][:, common]
dm = dm_all[rows]
ok = ~np.isnan(Xc).any(axis=1)
Xc, food, dm = Xc[ok], food[ok], dm[ok]

results={}
results["n_samples"]=int(Xc.shape[0])
results["n_common_bands"]=int(common.sum())
results["common_min"]=round(float(wlc.min()),0)
results["common_max"]=round(float(wlc.max()),0)
results["full_max"]=round(float(wl.max()),0)
results["foods"]={FOOD_ZH[f]:int((food==f).sum()) for f in ORDER}
print(f"Common-band matrix: {Xc.shape[0]} samples x {Xc.shape[1]} bands ({wlc.min():.0f}-{wlc.max():.0f} nm)")
print("Foods:", results["foods"])

def snv(a): return (a-a.mean(1,keepdims=True))/a.std(1,keepdims=True)
Xsnv=snv(Xc)

# ============ FIG 1: spectral signatures ============
fig,axes=plt.subplots(1,2,figsize=(12,4.6))
for f in ORDER:
    m=food==f; mu=Xc[m].mean(0); sd=Xc[m].std(0)
    axes[0].plot(wlc,mu,color=FOOD_COLOR[f],lw=2,label=FOOD_ZH[f])
    axes[0].fill_between(wlc,mu-sd,mu+sd,color=FOOD_COLOR[f],alpha=0.12)
axes[0].axvspan(398,700,color="#fde047",alpha=0.10); axes[0].axvspan(700,773,color="#fca5a5",alpha=0.12)
axes[0].set_title("原始反射光譜（平均 ± 標準差）",weight="bold")
axes[0].set_xlabel("波長 (nm)"); axes[0].set_ylabel("反射率"); axes[0].legend(fontsize=11,loc="upper left")
axes[0].text(545,axes[0].get_ylim()[1]*0.05,"可見光",fontsize=10,color="#a16207",ha="center")
axes[0].text(737,axes[0].get_ylim()[1]*0.05,"紅邊",fontsize=9,color="#b91c1c",ha="center")
for f in ORDER:
    m=food==f; axes[1].plot(wlc,Xsnv[m].mean(0),color=FOOD_COLOR[f],lw=2,label=FOOD_ZH[f])
axes[1].set_title("SNV 標準化後（消除散射差異）",weight="bold")
axes[1].set_xlabel("波長 (nm)"); axes[1].set_ylabel("SNV 反射率"); axes[1].legend(fontsize=11)
plt.tight_layout(); plt.savefig(IMG/"fig1_spectra.png",dpi=150); plt.close()
print("[OK] fig1_spectra.png")

# ============ FIG 2: PCA ============
pca=PCA(n_components=10).fit(Xsnv); sc=pca.transform(Xsnv); ev=pca.explained_variance_ratio_
results["pca_pc1"]=round(float(ev[0]*100),1); results["pca_pc2"]=round(float(ev[1]*100),1)
results["pca_pc12_cum"]=round(float(ev[:2].sum()*100),1)
fig,axes=plt.subplots(1,2,figsize=(12,4.6))
axes[0].bar(range(1,11),ev*100,color=TEAL,alpha=0.8)
axes[0].plot(range(1,11),np.cumsum(ev)*100,color=CORAL,marker="o",lw=2)
axes[0].set_title("PCA 解釋變異量",weight="bold")
axes[0].set_xlabel("主成分"); axes[0].set_ylabel("變異量 (%)")
axes[0].text(3.2,np.cumsum(ev)[1]*100-8,f"前2個PC = {results['pca_pc12_cum']}%",fontsize=11,color=CORAL)
for f in ORDER:
    m=food==f; axes[1].scatter(sc[m,0],sc[m,1],s=16,color=FOOD_COLOR[f],alpha=0.6,label=FOOD_ZH[f],edgecolors="none")
axes[1].set_title("PCA 散布圖：四種食材自然分群",weight="bold")
axes[1].set_xlabel(f"PC1 ({results['pca_pc1']}%)"); axes[1].set_ylabel(f"PC2 ({results['pca_pc2']}%)"); axes[1].legend(fontsize=11)
plt.tight_layout(); plt.savefig(IMG/"fig2_pca.png",dpi=150); plt.close()
print(f"[OK] fig2_pca.png PC1={results['pca_pc1']}% PC2={results['pca_pc2']}%")

# ============ FIG 3: classification ============
Xtr,Xte,ytr,yte=train_test_split(Xsnv,food,test_size=0.25,random_state=42,stratify=food)
svm=make_pipeline(StandardScaler(),SVC(kernel="rbf",C=10,gamma="scale")).fit(Xtr,ytr)
pred=svm.predict(Xte); acc=accuracy_score(yte,pred)
rf=RandomForestClassifier(n_estimators=300,random_state=42).fit(Xtr,ytr)
results["svm_accuracy"]=round(float(acc*100),1)
results["rf_accuracy"]=round(float(accuracy_score(yte,rf.predict(Xte))*100),1)
results["n_test"]=int(len(yte))
cm=confusion_matrix(yte,pred,labels=ORDER)
fig,ax=plt.subplots(figsize=(6.2,5.6))
im=ax.imshow(cm,cmap="BuGn")
ax.set_xticks(range(4)); ax.set_yticks(range(4))
ax.set_xticklabels([FOOD_ZH[l] for l in ORDER]); ax.set_yticklabels([FOOD_ZH[l] for l in ORDER])
ax.set_xlabel("預測"); ax.set_ylabel("實際")
ax.set_title(f"SVM 分類混淆矩陣（測試集 n={len(yte)}）\n準確率 = {results['svm_accuracy']}%",weight="bold")
for i in range(4):
    for j in range(4):
        ax.text(j,i,cm[i,j],ha="center",va="center",color="white" if cm[i,j]>cm.max()/2 else INK,fontsize=14,weight="bold")
plt.tight_layout(); plt.savefig(IMG/"fig3_confusion.png",dpi=150); plt.close()
print(f"[OK] fig3_confusion.png SVM={results['svm_accuracy']}% RF={results['rf_accuracy']}%")

# ============ FIG 4: dry-matter regression — LEEK full Vis-NIR ============
# leek has full 398-1717 nm; NIR is where composition/moisture lives
mL=(pref=="L").values
XL=Xall[mL]; dmL=dm_all[mL]
validL=~np.isnan(XL).any(axis=0)
XLv=XL[:,validL]; wlL=wl[validL]
okL=(~np.isnan(XLv).any(axis=1))&(~np.isnan(dmL))
XLv,dmL=XLv[okL],dmL[okL]*100
XLs=snv(XLv)
nc=12
pls=PLSRegression(n_components=nc)
yp=cross_val_predict(pls,XLs,dmL,cv=10).ravel()
r2=r2_score(dmL,yp); rmse=np.sqrt(mean_squared_error(dmL,yp))
results["pls_r2"]=round(float(r2),3); results["pls_rmse"]=round(float(rmse),2)
results["pls_ncomp"]=nc; results["dm_n"]=int(len(dmL))
results["leek_bands"]=int(validL.sum()); results["leek_max_nm"]=round(float(wlL.max()),0)
fig,ax=plt.subplots(figsize=(6.0,5.6))
ax.scatter(dmL,yp,s=20,color=GOLD,alpha=0.6,edgecolors="none",label="韭蔥樣本")
lim=[min(dmL.min(),yp.min())-1,max(dmL.max(),yp.max())+1]
ax.plot(lim,lim,"--",color=INK,lw=1.2,alpha=0.7,label="理想 1:1 線")
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_title(f"PLS 迴歸預測乾物質（韭蔥, {validL.sum()} 波段至 {wlL.max():.0f}nm）\n10-fold CV: R²={r2:.3f}, RMSE={rmse:.2f}%",weight="bold")
ax.set_xlabel("實際乾物質 (%)"); ax.set_ylabel("預測乾物質 (%)"); ax.legend(fontsize=11)
plt.tight_layout(); plt.savefig(IMG/"fig4_regression.png",dpi=150); plt.close()
print(f"[OK] fig4_regression.png PLS R²={r2:.3f} RMSE={rmse:.2f}%")

json.dump(results,open(ROOT/"results.json","w",encoding="utf-8"),ensure_ascii=False,indent=2)
print("\n=== RESULTS ==="); print(json.dumps(results,ensure_ascii=False,indent=2))
