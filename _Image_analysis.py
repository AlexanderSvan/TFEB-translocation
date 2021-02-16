import nd2reader as nd
import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import regionprops
import math
import scipy.ndimage as ndi
from skimage.filters import gaussian, threshold_otsu
from skimage.morphology import binary_erosion,disk, binary_closing
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
import seaborn as sns
import pandas as pd

class image:  
    def __init__(self, img=None, verbose=False):
        self.img=img
        self.original=img
        self.passed=np.zeros_like(img)
        self.failed=np.zeros_like(img)
        self.verbose=verbose
        self.stats={'passed': 0,'failed':0,'filtered':0}
        if img is not None:
            self.shape=img.shape

class nuc_segment(image):

    def blur(self, method='gaussian', kernel=1):
        if method=='gaussian':
            self.img=gaussian(self.img, sigma=kernel)
            self.post()
        
    def binary(self, param=None, size=1):
        if param is None:
            self.img=self.img>threshold_otsu(self.img)
        elif param.dtype==int:
            pass
        elif param =='closing':
            self.img=binary_closing(self.img, selem=disk(size))
        self.post()
    
    def post(self):
        if self.verbose==True:
            plt.imshow(self.img, interpolation='none')
            plt.show()
        else:
            pass
        
    def validate(self):
        passed=np.zeros_like(self.img)
        failed=np.zeros_like(self.img)
        self.failed=np.zeros_like(self.img)

        reg=regionprops(self.img)
        
        for obj in reg:
            if ((obj.major_axis_length/2)*(obj.minor_axis_length/2)*math.pi)>0:
                roundness=obj.area/((obj.major_axis_length/2)*(obj.minor_axis_length/2)*math.pi)
            else:
                continue
            if (roundness>0.98) & (roundness<1.02):
                passed=passed+(binary_erosion(self.img==obj.label)).astype(int)
            else:
                failed=failed+(self.img==obj.label).astype(int)
        self.passed=self.passed+passed
        self.failed=self.failed+failed
        self.img=self.failed

    def watershed(self):
        distance = ndi.distance_transform_edt(self.img)
        local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((8, 8)),
                                    labels=binary_erosion(self.img, selem=disk(5)))
        markers = ndi.label(local_maxi)[0]
        labels = watershed(-distance, markers, mask=self.img)
        self.img=labels
        self.post()
        
    def filter_size(self, low=None, high=None):
        
        if (low is None) & (high is None):
            print('Cutoff has not been set')
            return
        
        lab=ndi.label(self.img)[0]
        regions=regionprops(lab)
        canvas=np.zeros_like(self.img)
        for reg in regions:
            if low is not None:
                if high is not None:
                    if (reg.area>low) & (reg.area<high):
                        canvas=canvas+(lab==reg.label).astype(int)
                else:
                    if (reg.area>low):
                        canvas=canvas+(lab==reg.label).astype(int)
            elif high is not None:
                if reg.area<high:
                    canvas=canvas+(lab==reg.label).astype(int)
        self.stats['filtered']+=ndi.label(self.img)[1]-ndi.label(canvas)[1]
        self.img=canvas
    
    def plot(self):
        lab=ndi.label(self.img)[0]
        regions=regionprops(lab)
        
        sizes=[obj.area for obj in regions]
        if self.verbose==True:
            sns.swarmplot(sizes)
            plt.show()
    
    def label(self):
        self.img=ndi.label(self.img)[0]
        self.post()
    
    def save(self):
        self.img=self.passed
        self.all_nuc=((self.failed+self.img)>0).astype(int)

    
    def reset(self):
        self.img=self.original
        self.post()
        
    def summary(self, verbose=False):
        self.stats['passed']=ndi.label(self.img)[1]
        self.stats['failed']=ndi.label(self.failed)[1]
        if verbose==True:
            print(self.stats)

class cellAnalyser:
    
    def __init__(self, int_img, nuc_mask,nuc_number, cell_mask):
        self._int_img=int_img
        self.nuc=nuc_mask>0
        self._cell_mask=cell_mask
        self.nuc_label=nuc_number
        
    @property
    def cell_label(self):
        return np.bincount(self._cell_mask[self.nuc]).argmax()
    
    @property
    def cytosol_mask(self):
        return (self._cell_mask==self.cell_label.astype(int)-self.nuc.astype(int))>0

    @property
    def nuc_int(self):
        return np.mean(self._int_img[self.nuc])
    
    @property
    def cell_int(self):
        return np.mean(self._int_img[self._cell_mask==self.cell_label.astype(int)])
    
    @property
    def cytosol_int(self):
        return np.mean(self._int_img[self.cytosol_mask])
    
    @property
    def trans_ratio(self):
        return self.nuc_int/self.cytosol_int
    
class cell_segmentation:
    
    def __init__(self, nuclei, cells):
        self.nuc=nuclei
        self.cells=cells
    
    def blur(self, sigma=1):
        self.cells=gaussian(self.cells, sigma=sigma)
        
    def mask(self, threshold=None):
        if threshold==None:
            threshold=np.percentile(self.cells, 20)
        self.mask=self.cells>threshold
        
    def gapfill(self, size=4):
        self.mask=binary_closing(self.mask, selem=disk(size))
        
    def watershed(self):
        self.segm=watershed(-self.nuc, mask=self.mask)
        
def analyser(int_img, nuclei, cells):
    
    res=[]
    labels=ndi.label(nuclei)[0]
    for a in range(1,np.amax(labels)):
        nuc_mask=labels==a
        nuclei=labels*nuc_mask.astype(int)
        res.append(cellAnalyser(int_img, nuclei, a, cells))
    return res

#%% Quantification of translocation
def sample_analysis(img):
    results=[]
    for v in range(img.sizes['v']):
        img.default_coords['v']=v
        
        nuc=nuc_segment(img=img[0], verbose=False)
        nuc.post()
        nuc.blur(kernel=3)
        nuc.binary()
        nuc.label()
        nuc.validate()
        nuc.post()
        nuc.watershed()
        nuc.validate()
        nuc.save()
        nuc.plot()
        nuc.filter_size(low=300,high=1500)
        nuc.plot()
        nuc.post()
        nuc.summary()
        
        mask=cell_segmentation(nuc.all_nuc, img[1])
        mask.blur()
        mask.mask(0.05)
        mask.gapfill()
        mask.watershed()
        a=analyser(img[1], nuc.img, mask.segm)
        results.append(np.mean([obj.trans_ratio for obj in a]))
    return results


import os
import pandas as pd
import itertools
path='TFEB/'


data={}
for file in [file for file in os.listdir(path) if ".nd2" in file]:
    img=nd.ND2Reader(path+file)
    
    img.sizes
    
    img.bundle_axes='yx'
    img.iter_axes='c'
    img.default_coords['v']=0

    print(file)

    data[file]=sample_analysis(img)
translocation=pd.DataFrame(dict([ (k,pd.Series(np.array(v))) for k,v in data.items()]))
translocation.index=list(itertools.chain(*[[a,a,a,a,a] for a in ['well_1','well_2','well_3']]))
translocation.columns=translocation.columns.str.split('.').str[0]


translocation.to_csv(path+'translocation.csv')

    
#%% General intensity
def sample_analysis(img):
    results=[]
    for v in range(img.sizes['v']):
        img.default_coords['v']=v
        
        nuc=nuc_segment(img=img[0], verbose=False)
        nuc.post()
        nuc.blur(kernel=3)
        nuc.binary()
        nuc.label()
        nuc.validate()
        nuc.post()
        nuc.watershed()
        nuc.validate()
        nuc.save()
        nuc.plot()
        nuc.filter_size(low=300,high=1500)
        nuc.plot()
        nuc.post()
        nuc.summary()
        
        mask=cell_segmentation(nuc.all_nuc, img[1])
        mask.blur()
        mask.mask(0.05)
        mask.gapfill()
        mask.watershed()
       
        a=analyser(img[1], nuc.img, mask.segm)
        results.append(np.mean([obj.cell_int for obj in a]))
    return results


import os
import pandas as pd
path='TFEB/'


data={}
for file in [file for file in os.listdir(path) if ".nd2" in file]:
    img=nd.ND2Reader(path+file)
    
    img.sizes
    
    img.bundle_axes='yx'
    img.iter_axes='c'
    img.default_coords['v']=0

    print(file)

    data[file.split('.')[0]]=sample_analysis(img)
df_int=pd.DataFrame(dict([ (k,pd.Series(np.array(v))) for k,v in data.items()]))
df_int.index=list(itertools.chain(*[[a,a,a,a,a] for a in ['well_1','well_2','well_3']]))
df_int.columns=df_int.columns.str.split('.').str[0]

df_int.to_csv(path+'mean_int.csv')

    
#%%
import seaborn as sns

font = {'family' : 'normal',
        'weight' : 'regular',
        'size'   : 15}

plt.rc('font', **font)


translocation=pd.read_csv(path+'translocation.csv', index_col=0)

trans_data=translocation[['Veh','Enza','GF','SB80','SB90','VX']].T
trans_data=trans_data.T.reset_index().groupby('index').mean()

pd.DataFrame(trans_data.melt()).T.to_csv(path+'translocation_means_for_stats.csv', index=None, header=None)

# plt.figure(figsize=(3,5), dpi=300)
fig, ax=plt.subplots(figsize=(3,5), dpi=300)
# sns.boxplot(x='variable', y='value', data=trans_data.melt(), width=0.4)
# sns.barplot(x='variable', y='value', data=trans_data.melt())
ax.bar(np.arange(len(trans_data.columns)), trans_data.mean(axis=0), edgecolor ='black', linewidth=1.5 ,
       width=0.7, yerr=trans_data.std(axis=0), capsize=4)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_ylabel('TFEB nucleus-cytoplasm \n ratio')
# sns.swarmplot(x='variable', y='value', data=trans_data.T.melt())
plt.xticks(np.arange(len(trans_data.columns)),trans_data.columns, rotation=45)
plt.tight_layout()
plt.savefig(path+'translocation_barchart.eps', dpi=300)

#%%

df_int=pd.read_csv(path+'mean_int.csv', index_col=0)
meanInt_data=df_int[['Veh','Enza','GF','SB80','SB90','VX']].T
meanInt_data=meanInt_data.T.reset_index().groupby('index').mean()

# plt.figure(figsize=(3,5), dpi=300)
fig, ax=plt.subplots(figsize=(3,5), dpi=300)
# sns.boxplot(x='variable', y='value', data=meanInt_data.melt(), width=0.4)
# sns.barplot(x='variable', y='value', data=meanInt_data.melt())
# sns.barplot(x='variable', y='value', data=trans_data.melt())
ax.bar(np.arange(len(meanInt_data.columns)), meanInt_data.mean(axis=0), edgecolor ='black', linewidth=1.5 ,
       width=0.7, yerr=meanInt_data.std(axis=0), capsize=4)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_ylabel('TFEB mean fluorescence \n (A.U.)')
plt.xticks(np.arange(len(meanInt_data.columns)),meanInt_data.columns, rotation=45)

