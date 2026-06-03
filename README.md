# 高光譜資料分析：Python + Orange Data Mining（食品篇）
# Hyperspectral Data Analysis with Python + Orange Data Mining (Food)

互動教學投影片（中／英）+ 可重現的 Python 分析腳本 + Orange 視覺化工作流。
針對食品化學研究生，使用真實開源資料集 **SpectroFood**（Zenodo 8362947）。

## 🌐 線上瀏覽 / Live site
GitHub Pages：開啟 `index.html`（教材中心），或直接看：
- 中文投影片：`zh/index.html`
- English slides：`en/index.html`

> 投影片用鍵盤 **← →** 翻頁，**F** 全螢幕。

## 內容 / Contents
| 檔案 | 說明 |
|------|------|
| `zh/`, `en/` | 9 頁互動投影片（資料立方體 → 光譜指紋 → SNV → PCA → 分類 → PLS 迴歸 → Orange 工作流） |
| `analysis/run_analysis.py` | Python (scikit-learn) 完整分析，產生所有圖表 + `results.json` |
| `analysis/run_analysis_en.py` | 同分析，英文標籤圖表 |
| `analysis/spectrofood_workflow.ows` | Orange Data Mining 工作流（File → Preprocess Spectra → PCA / PLS / SVM → Test & Score → Confusion Matrix） |
| `analysis/results.json` | 鎖定的真實數值 |

## 真實結果 / Verified results
- 1028 樣本、4 食材、141 共同 Vis 波段（398–773nm）
- PCA：前 2 主成分 = **94.6%** 變異
- 分類（SVM / Random Forest）：測試集 **100%**
- PLS 迴歸乾物質（韭蔥，NIR 至 1717nm）：**R²=0.840、RMSE=0.96%**

## 重現 / Reproduce
```bash
pip install spectral scikit-learn scipy numpy matplotlib pandas
python analysis/run_analysis.py    # 會自動下載 SpectroFood CSV
```
Orange：`pip install orange3 orange-spectroscopy`，開啟 `analysis/spectrofood_workflow.ows`。

## 資料來源 / Data source
SpectroFood dataset — https://zenodo.org/records/8362947
