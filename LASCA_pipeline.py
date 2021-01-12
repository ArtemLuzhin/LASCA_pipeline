
# coding: utf-8

# In[1]:


from __future__ import division
import numpy as np
import scipy
import pandas as pd
import statsmodels.stats.multitest
from sklearn.cluster import DBSCAN
#from scipy import sparse
import cooler


# ## Main functions

# In[9]:


def weibul_func(x,arg1,arg2,arg3):
    out=scipy.stats.weibull_min.pdf(x,arg1,arg2,arg3)
    return(out)


def convert_to_log10_square_matrix(q_vals_array):
    """convert q-values array to log10(q-value) square matrix"""
    log=np.log10(np.array(q_vals_array))
    zeros=np.zeros((np.array(q_vals_array).shape[0],np.array(q_vals_array).shape[0]+np.array(q_vals_array).shape[1]))
    for i in range(0,np.array(q_vals_array).shape[0]+1):
        zeros[i:i+1,i:i+np.array(q_vals_array).shape[1]]=log[i:i+1,:]
    return(zeros)

#latest verstion of caller
def Get_pvalue_v7(raw_mtx,mtx,resolution,first_anchor_coordinate,last_anchor_coordinate,distance_bp,FDR=0.05,bin_coverage=0.25,q=0.95):
    """Function permorm fitting of Weibull distribution in data, calculate p-values,
    adjust p-values by FDR correction and return array with q-values
    raw_mtx - HiC matrix used to calculate coverage
    mtx - corrected HiC matrix
    resolution - HiC resolution in bp
    first_anchor_coordinate,last_anchor_coordinate - coordinates in bp of region to analyze.
    To analyze full chromosome set to 0 and to length of chromosome
    distance_bp - upper threshold distance in bp to search interactions, 
    should be set to biological relevant value
    FDR - FDR correction, default 0.05
    bin_coverage - mean coverage of diagonal of matrix to take into analysis default: 0.25
    q - set up quantile value, default: 0.95, this value seting up model strength
    return q-values array"""
    #start_bin=int(ROI_start/resolution)
    #end_bin=int(ROI_end/resolution) # one should be discuss!!!!!!!!!
    first_anchor_bin=int(first_anchor_coordinate/resolution)
    last_anchor_bin=int(last_anchor_coordinate/resolution)
    #initialize NaN's around ROI
    #mtx[:start_bin,:start_bin]=np.nan
    #mtx[end_bin+1:,end_bin+1:]=np.nan
    #mtx[end_bin+1:,:start_bin]=np.nan
    #mtx[:start_bin,end_bin+1:]=np.nan
    #fill main diagonal with zeros
    np.fill_diagonal(mtx,0)
    
    #initialize NaN's around ROI
    #raw_mtx[:start_bin,:start_bin]=np.nan
    #raw_mtx[end_bin+1:,end_bin+1:]=np.nan
    #raw_mtx[end_bin+1:,:start_bin]=np.nan
    #raw_mtx[:start_bin,end_bin+1:]=np.nan
    #fill main diagonal with zeros
    np.fill_diagonal(raw_mtx,0)
    
    # Calculate coverage
    #drod all values in diagonal that >=0.95% quantile
    print('Calculating coverage per bin...')
    coverage=[]
    for d in range(1,distance_bp//resolution,1):
        diag=np.diag(raw_mtx,k=d)
        #drop NaN's
        diag=diag[~np.isnan(diag)]
        # drop elements>95% quantile
        diag=diag[diag<np.nanquantile(diag,q)]
        coverage.append(np.nanmean(diag))
    print('Filter poor bins with coverage <=',bin_coverage)
    diags=np.arange(1,distance_bp//resolution,1)
    filtered_diags=np.array([],dtype=int)
    for el1,el2 in zip(diags,coverage):
        if el2>=bin_coverage:
            filtered_diags=np.append(filtered_diags,el1)
    print('Filtering complete.')
    print('Start fitting Weibull distribution...')
    Weibul_parameters=[]
    for el in filtered_diags:
        diag=np.diag(mtx,k=int(el))
        #drop zeros
        diag=diag[diag>0]
        #drop NaN's
        diag=diag[~np.isnan(diag)]
        # drop elements>95% quantile
        diag=diag[diag<np.quantile(diag,q)]
        # fit Weibul
        args=scipy.stats.weibull_min.fit(np.sort(diag),floc=0)
        Weibul_parameters.append(args) # list with arguments for selected diagonals (filtered_diags)
    print('Distribution fitted.')
    #Now we have array of filtered diags number and Weibull parameters for it
    # iterate over anchor bins
    print('Calculate q-values...')
    out=[]
    for anchor_bin in range(first_anchor_bin,last_anchor_bin+1):
        # right interactions
        #RI=mtx[anchor_bin:anchor_bin+1,anchor_bin:][0] # this include main diagonal so RI[0] is a anchor point itself
        #interactions by distance
        RI=mtx[anchor_bin:anchor_bin+1,anchor_bin:anchor_bin+(distance_bp//resolution)][0] # this include main diagonal so RI[0] is a anchor point itself
        #check number of interactions if less than distance add zeros to the end
        if len(RI)<(distance_bp//resolution):
            RI=np.append(RI,np.zeros((distance_bp//resolution)-len(RI)))
        RI_pvalues=[]
        #left interactions
        #LI=mtx[anchor_bin:anchor_bin+1,:anchor_bin][0] #not include main diagonal point
        #interactions by distance
        LI_pvalues=[]
        LI=mtx[anchor_bin:anchor_bin+1,anchor_bin-(distance_bp//resolution):anchor_bin][0] #not include main diagonal point
        #check number of interactions if less than distance add zeros to the start
        if len(LI)<(distance_bp//resolution):
            LI=np.append(np.zeros((distance_bp//resolution)-len(LI)),LI)
        for d,p in zip(filtered_diags,Weibul_parameters):
            # integrate
            #tail probability using Weibull CDF
            RI_pvalues.append(1-scipy.stats.weibull_min.cdf(RI[d],p[0],p[1],p[2]))
            LI_pvalues.append(1-scipy.stats.weibull_min.cdf(LI[-d],p[0],p[1],p[2]))
        #LI_pvalues.reverse()
        #adjust pvalues
        LI_qvalues=statsmodels.stats.multitest.fdrcorrection(LI_pvalues,alpha=FDR)[1]
        RI_qvalues=statsmodels.stats.multitest.fdrcorrection(RI_pvalues,alpha=FDR)[1]
        # reassign q values to original bin position
        R=np.ones(len(RI)) #inclide main diagonal point R[0]
        L=np.ones(len(LI))
        R[filtered_diags]=RI_qvalues
        L[np.array(filtered_diags)*-1]=LI_qvalues
        out.append(np.concatenate((L,R)))
    print('Finished!')
    return(out)


def Get_qvals_mtx_v2(qvals_result,distance_bins):
    """makes q-values array, used for clusterisation"""
    return(qvals_result[:,distance_bins:qvals_result.shape[1]-distance_bins])


def cluster_dots(qvals_mtx,mtx,min_cluster_size=2,q_value_treshold=0.01,filter_by_coverage=True,min_coverage=0,filter_dist=1):
    """qvals_mtx-matrix of q_values produced by Get_qvals_mtx_v2 function
       mtx- corrected HiC matrix
       min_cluster_size - minimun size of posible loop in dots
       p_value_treshold - minimum q value to assign dot as significant
       filter_by_coverage if true then algorithm filter out dots near low coverage bins by user distance in bins
       min_coverage - used when filter_by_coverage == True, min sum in bin to select as high coverege 
       (calculated on normalized matrix)
       filter_dist - used when filter_by_coverage == True, +/- distance around low coverage bin
       
       Function perform clustering of significant dots and return 
       table with x,y coordinates, intensity (# of HiC contacts) and cluster label.
       """
    #significant dots coordinates
    dots=np.where((qvals_mtx<=np.log10(q_value_treshold)) & (qvals_mtx!=-np.inf) & (qvals_mtx!=np.isnan(qvals_mtx)))
    #intensity values
    I=[]
    for x,y in zip(dots[0],dots[1]):
        I.append(mtx[x,y])
    I=np.array(I)
    data=np.concatenate((np.vstack(dots[0]),np.vstack(dots[1]),np.vstack(I)),axis=1)
    #cluster adjuscent dots
    clustering = DBSCAN(eps=1.0, min_samples=min_cluster_size).fit(data[:,0:2])
    data=np.concatenate((data,np.vstack(clustering.labels_)),axis=1)
    #filter noise dots
    data=data[data[:,3]!=-1]
    #reorder first two values in row such that 0<1
    for el,i in zip(data,[x for x in range(len(data))]):
        if el[0]>el[1]:
            data[i]=el[[1,0,2,3]]
    
    #create temporary pandas dataframe
    tmp_df=pd.DataFrame(data)
    #group by coordinates and sum intensity
    tmp_df=tmp_df.groupby([0,1],as_index=False).sum()
    #filter by coverage
    if filter_by_coverage==True:
        #filter dots near low coverage bins
        coordinates=np.where(np.sum(mtx,axis=1)<=min_coverage)[0]
        out=np.array([])
        for i in range(1,filter_dist+1):
            plus=coordinates+i
            minus=coordinates-i
            new_coordinates=np.concatenate((plus,minus))
            out=np.concatenate((out,new_coordinates))
        coordinates=np.concatenate((coordinates,out))
        
        tmp_df=tmp_df[(~tmp_df[0].isin(coordinates))]
        tmp_df=tmp_df[(~tmp_df[1].isin(coordinates))]
        tmp_df=tmp_df.groupby(3).filter(lambda x: len(x) >= min_cluster_size)
    
    #reclaster data
    clustering = DBSCAN(eps=1.0, min_samples=min_cluster_size).fit(tmp_df[[0,1]])
    #set new label
    tmp_df[3]=clustering.labels_
    #filter noise
    tmp_df=tmp_df[tmp_df[3]!=-1]
    return(tmp_df)


def Get_coordinates(df,Intensity=False,as_intervals=True):
    """Input: dataframe returned by cluster_dots 
    If Intensity set to True return brightest dot, else return by centroid coordinates
    set as_intervals to True to export loop coordinates as bins range
    return table with coordinates in BINS!!!"""
    if Intensity==True:
        df.sort_values(2,inplace=True)
        df=df.drop_duplicates(3,keep='last').sort_values(3)
    else:
        df=df.groupby(by=3,as_index=False).mean()
    
    df=df[[0,1,3]]
    df.columns=['start','end','ID']
    
    if as_intervals==False:
        return(df.astype('int'))
    else:
        int_df=pd.DataFrame()
        int_df['start1']=df['start']
        int_df['start2']=df['start']+1
        
        int_df['ID']=df['ID'] # move ID
        int_df['end1']=df['end']
        int_df['end2']=df['end']+1
        
        return(int_df.astype('int'))


def get_scaling(mtx,bin_start,bin_end):
    """return contact probability normalized by maximum of matrix
    mtx - HiC matrix
    bin_start,bin_end - number of start and end diagonals"""
    out=[]
    for d in range(bin_start,bin_end,1):
        out.append(np.sum(np.diag(mtx,d)))
    return(np.array(out)/np.max(out))


def filter_qvalues_by_scaling(qvals_mtx,scale_params,q_value_treshold=0.01):
    """adjust q-values by scaling factor, 
    sometimes can improve results in high noise/high resolution matrix.
    qvals_mtx - q-values matrix from Get_qvals_mtx_v2 function
    scale_params - result from get_scaling function
    q_value_treshold - q-value treshold 
    return adjusted by scaling q-values matrix"""
    tresholds=q_value_treshold*scale_params
    tresholds=np.log10(tresholds)
    #make zero matrix for output
    #z=np.zeros(qvals_mtx.shape)
    #iterating over diagonals
    rows, cols = np.indices(qvals_mtx.shape)
    for d in range(-len(tresholds)+1,len(tresholds),1):
        tmp_diag=np.diag(qvals_mtx,d).copy()
        coef=tresholds[abs(d)]
        #fill not significant with zeros
        tmp_diag[tmp_diag >= coef] = 0
        #fill particular diagonal
        #index value array
        #rows, cols = np.indices(qvals_mtx.shape)
        row_vals = np.diag(rows, k=d)
        col_vals = np.diag(cols, k=d)
        #z[row_vals, col_vals]=tmp_diag
        qvals_mtx[row_vals, col_vals]=tmp_diag
    #return(z)
    return(qvals_mtx)


def Export_as_bedpe(coord,chrm,resolution,outfile):
    """convert coordinate table from Get_coordinates of dot_filter functions  to bedpe format
    and save it 
    coord - coordinate table
    chrm - chromosome name
    resolution - HiC resolution
    outfile - name of output bedpe"""
    out=pd.DataFrame()
    coord['chrom']=chrm
    #make chrom first
    cols = coord.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    #reorder
    coord = coord[cols]
    #concatenate result
    out=pd.concat([out,coord])
    out['start1']=out['start1']*resolution
    out['start2']=out['start2']*resolution
    out['end1']=out['end1']*resolution
    out['end2']=out['end2']*resolution
    out['ID']=chrm
    out['.0']='.'
    out['.1']='.'
    out['.2']='.'
    out['.3']='.'
    out['color']='0,255,0'
    out.to_csv(outfile,header=False,index=False,sep='\t')
    return('Saved: '+ str(outfile))


def dot_filter_v5(coord,mtx):
    """Additional filter, usefull for human data
    return intensity of central pixel,
    number of zeros around center,
    difference between 8 neighbors and 24 neighbors for decay analysis,
    PA score and APA score
    coord - coordinates from Get_coordinates function 
    mtx - corrected HiC matrix"""
    int_1=[]
    zeros=[]
    dif_1=[]
    dif_2=[]
    PA=[]
    APA=[]
    #mtx=mtx/np.nansum(mtx)
    for a,b in coord[['start1','end1']].values:
        if len(mtx[a-1:a+2,b-1:b+2].copy())==0 or len(mtx[a-2:a+3,b-2:b+3].copy())==0 or len(mtx[a-5:a+6,b-5:b+6].copy())==0:
            APA.append(0)
            PA.append(0)
            int_1.append(mtx[a,b].copy()) #Intensity of center
            zeros.append(0)
            dif_1.append(0)
            dif_2.append(0)
            continue
        center=mtx[a,b].copy()
        round_1=mtx[a-1:a+2,b-1:b+2].copy() #8 pixels
        round_1[1,1]=np.nan #mask center
        
        round_2=mtx[a-2:a+3,b-2:b+3].copy() #24 pixels
        round_2[1:4,1:4]=np.nan #mask center + 8 pixels
        
        round_3=mtx[a-5:a+6,b-5:b+6].copy() # sqaure 11x11 around loop center
        corner_1=np.nanmean(round_3[7:,:3])
        corner_2=np.nanmean(round_3[0:3,0:3])
        corner_3=np.nanmean(round_3[0:3,8:])
        corner_4=np.nanmean(round_3[8:,8:])
        dif_1.append(np.nanmean(center-round_1.flatten()))
        dif_2.append(np.nanmean(center-round_2.flatten()))
        
        tmp=np.concatenate((round_1.flatten(),round_2.flatten()))
        z_num=len(np.where([tmp==0])[0]) # number of zeros
        
        corner=np.mean([corner_1,corner_2,corner_3,corner_4])
        APA.append(center/corner)
        PA.append(np.nanmean(round_2)/corner)
        int_1.append(center) #Intensity of center
        zeros.append(z_num)
    
    
    coord['INT1']=np.array(int_1)
    coord['DIF1']=np.array(dif_1)
    coord['DIF2']=np.array(dif_2)
    coord['PA']=np.array(PA)
    coord['APA']=np.array(APA)
    coord['ZEROS']=np.array(zeros)
    return(coord)


# ## Processing functions

# In[10]:


def LASCA_processing(raw_mtx,mtx,chr_name,output_bedpe_name,resolution,start_bin,end_bin,distance_bins,FDR=0.1,q=0.90,                     adjust_by_scale=False,scaling_q_value_trhd=0.05,min_cluster=3,                     filter_bins=2,q_value_trhd=0.1,Intensity=True,as_intervals=True,bin_coverage=0.25,                     save_qvalues_mtx=False,use_filter=False, filter_zeros=2, filter_PA=1,filter_APA=1.9,                    filter_intensity=0.3):
    """main procesing function, for parameter description see article."""
    np.nan_to_num(mtx,copy=False)
    np.nan_to_num(raw_mtx,copy=False)
    q_vals_full_chrom=Get_pvalue_v7(raw_mtx,mtx,resolution,start_bin*resolution,                                    end_bin*resolution,distance_bins*resolution,FDR=FDR,q=q)
    #convert q-values to log10
    result=convert_to_log10_square_matrix(q_vals_full_chrom)
    qvals_mtx=Get_qvals_mtx_v2(result,distance_bins)
    
    if adjust_by_scale==True:
        qvals_by_scale=filter_qvalues_by_scaling(qvals_mtx,get_scaling(mtx,0,distance_bins),scaling_q_value_trhd)
        qvals_mtx=qvals_by_scale
    else:
        qvals_mtx=qvals_mtx
    
    tmp_df=cluster_dots(qvals_mtx,mtx,min_cluster_size=min_cluster,filter_dist=filter_bins,                        q_value_treshold=q_value_trhd)
    coord=Get_coordinates(tmp_df,Intensity=Intensity,as_intervals=as_intervals)
    
    
    # save q-values matrix, can be used for fast adjustment later
    
    if save_qvalues_mtx==True:
        #np.save(output_bedpe_name[:-5]+'npy',qvals_mtx)
        np.savez_compressed(output_bedpe_name[:-6],qvals=qvals_mtx)
    else:
        print('q-value matrix not saved')
    
    if use_filter==True:
        tmp_coord=dot_filter_v5(coord,mtx)
        tmp_coord=tmp_coord[(tmp_coord['DIF1']>0) & (tmp_coord['DIF1']<tmp_coord['DIF2'])                             & (tmp_coord['ZEROS']<filter_zeros) & (tmp_coord['PA']>filter_PA)                             & (tmp_coord['APA']>filter_APA)                            & (tmp_coord['INT1']>np.quantile(tmp_coord['INT1'],filter_intensity))]
        coord=tmp_coord
    else:
        coord=coord
    
    Export_as_bedpe(coord,chr_name,resolution,output_bedpe_name)
    print('Finished!')
    

def LASCA_parameters_adjust(q_vals_file,mtx,chr_name,output_bedpe_name,resolution,distance_bins,                            adjust_by_scale=False,scaling_q_value_trhd=0.05,min_cluster=3,                            filter_bins=2,q_value_trhd=0.1,Intensity=True,as_intervals=True,                            use_filter=False, filter_zeros=2, filter_PA=1,filter_APA=1.9,                            filter_intensity=0.3):
    """This function can be used for fast parameter adjustment. 
    If in LASCA_processing save_qvalues_mtx=True
    for parameter description see article."""
    np.nan_to_num(mtx,copy=False)
    
    #convert q-values to log10
    #result=convert_to_log10_square_matrix(q_vals_full_chrom)
    tmp_qvals=np.load(q_vals_file)
    qvals_mtx=tmp_qvals['qvals']
    
    if adjust_by_scale==True:
        qvals_by_scale=filter_qvalues_by_scaling(qvals_mtx,get_scaling(mtx,0,distance_bins),scaling_q_value_trhd)
        qvals_mtx=qvals_by_scale
    else:
        qvals_mtx=qvals_mtx
    
    tmp_df=cluster_dots(qvals_mtx,mtx,min_cluster_size=min_cluster,filter_dist=filter_bins,                        q_value_treshold=q_value_trhd)
    coord=Get_coordinates(tmp_df,Intensity=Intensity,as_intervals=as_intervals)
    
    if use_filter==True:
        tmp_coord=dot_filter_v5(coord,mtx)
        tmp_coord=tmp_coord[(tmp_coord['DIF1']>0) & (tmp_coord['DIF1']<tmp_coord['DIF2'])                             & (tmp_coord['ZEROS']<filter_zeros) & (tmp_coord['PA']>filter_PA)                             & (tmp_coord['APA']>filter_APA)                            & (tmp_coord['INT1']>np.quantile(tmp_coord['INT1'],filter_intensity))]
        coord=tmp_coord
    else:
        coord=coord
    
    Export_as_bedpe(coord,chr_name,resolution,output_bedpe_name)
    print('Finished!')
    #return(tmp_df,coord)


# ## Utilites

# In[11]:


def mark_overlaps_low_res(low_list,high_list):
    """mark overlapping loops between low resolution loops 
    and high resolution loops,
    return lists of loops in which high resolution overlaping loops marked by 1"""
    marker=np.zeros(len(high_list))
    for index, row in low_list.iterrows():
        c1=row[0],row[1],row[2],row[4],row[5]
        for index2, row2 in high_list.iterrows():
            c2=row2[0],row2[1],row2[2],row2[4
                                           ],row2[5]
            if c2[0]==c1[0] and abs(c2[1]>=c1[1]) and c2[2]<=c1[2] and abs(c2[3]>=c1[3]) and c2[4]<=c1[4]:
                marker[index2]=marker[index2]+1
            else:
                marker[index2]=marker[index2]+0
    high_list[17]=marker
    low_list[17]=0
    return(high_list,low_list)

def mark_overlaps_high_res(low_list,high_list):
    """mark overlapping loops between low resolution loops 
    and high resolution loops,
    return lists of loops in which low resolution overlaping loops marked by 1"""
    
    marker=np.zeros(len(low_list))
    for index, row in low_list.iterrows():
        c1=row[0],row[1],row[2],row[4],row[5]
        for index2, row2 in high_list.iterrows():
            c2=row2[0],row2[1],row2[2],row2[4
                                           ],row2[5]
            if c2[0]==c1[0] and abs(c2[1]>=c1[1]) and c2[2]<=c1[2] and abs(c2[3]>=c1[3]) and c2[4]<=c1[4]:
                marker[index]=marker[index]+1
            else:
                marker[index]=marker[index]+0
    high_list[17]=0
    low_list[17]=marker
    return(high_list,low_list)

def merge_loop_list_low_res(loop_list):
    """merge provided bedpe list of loops using mark_overlaps_low_res function
    function drop high resolution overlapping loops.
    For example if 20Kb loop overlaps 5Kb loop, the 5Kb loop was dropped."""
    tmp=[]
    for el in loop_list:
        tmp.append(pd.read_csv(el,header=None,sep='\t'))
    #check resolution order
    resolutions=[]
    for i in range(len(tmp)):
        resolutions.append((tmp[i][:1][2].values-tmp[i][:1][1].values)[0])
    
    if resolutions!=list(np.sort(resolutions)):
        print("resolutions order is incorrect",resolutions)
        return()
    
    for i in range(1,len(resolutions)):
                   tmp[i-1],tmp[i]=mark_overlaps_low_res(tmp[i],tmp[i-1])
    df=pd.concat(tmp)
    return(df[df[17]==0].drop(17,axis=1))

def merge_loop_list_high_res(loop_list):
    """merge provided bedpe list of loops using mark_overlaps_high_res function
    function drop low resolution overlapping loops.
    For example if 20Kb loop overlaps 5Kb loop, the 20Kb loop was dropped."""
    tmp=[]
    for el in loop_list:
        tmp.append(pd.read_csv(el,header=None,sep='\t'))
    
    #check resolution order
    resolutions=[]
    for i in range(len(tmp)):
        resolutions.append((tmp[i][:1][2].values-tmp[i][:1][1].values)[0])
    
    if resolutions!=list(np.sort(resolutions)):
        print("resolutions order is incorrect",resolutions)
        return()
    
    for i in range(1,len(resolutions)):
                   tmp[i-1],tmp[i]=mark_overlaps_high_res(tmp[i],tmp[i-1])
    df=pd.concat(tmp)
    
    return(df[df[17]==0].drop(17,axis=1))


# In[12]:


def APA_analysis(a,b,mtx,return_PA=False):
    """
    a,b - start, end coordinates of loop in bins
    mtx - corrected HiC matrix"""
    PA=0
    APA=0
    #mtx=mtx/np.nansum(mtx)
    center=mtx[a,b].copy()
    round_1=mtx[a-1:a+2,b-1:b+2].copy() #8 pixels
    round_1[1,1]=np.nan #mask center
        
    round_2=mtx[a-2:a+3,b-2:b+3].copy() #24 pixels
    round_2[1:4,1:4]=np.nan #mask center + 8 pixels
    round_3=mtx[a-5:a+6,b-5:b+6].copy() # sqaure 11x11 around loop center
    corner_1=np.nanmean(round_3[7:,:3])
    corner_2=np.nanmean(round_3[0:3,0:3])
    corner_3=np.nanmean(round_3[0:3,8:])
    corner_4=np.nanmean(round_3[8:,8:])
    corner=np.mean([corner_1,corner_2,corner_3,corner_4])
    APA=center/corner
    PA=np.nanmean(round_2)/corner
    if return_PA==False:
        return(APA)
    else:
        return(PA)


# In[2]:


def APA_test_over_random(bedpe,coolfile,distance_bp,num,distance_initial_bp,resolution,chrom_name,PA=False,percentile=0.90):
    """bedpe - bedpe file with loops, pandas dataframe
    distance_bp - distance in bp from center of loop, used to receive random loops
    num - number of random loops
    distance_initial - starting distance from which to start selecting random loops
    resolution - HiC resolution
    chrom_name - chromosome name should be the same in bedpe and cool"""
    mtx=coolfile.matrix(balance=True).fetch(chrom_name)
    np.nan_to_num(mtx,copy=False)
    #size of mtx
    s=mtx.shape[0]
    print(s)
    loops_chrom=bedpe[bedpe[0]==chrom_name]
    loops_chrom[1]=loops_chrom[1]//resolution
    loops_chrom[4]=loops_chrom[4]//resolution
    #generate list of random distances in bins
    
    #pos=np.random.randint(low=distance_initial_bp//resolution, high=distance_bp//resolution, size=num, dtype='l')
    #neg=np.random.randint(low=-distance_bp//resolution, high=-distance_initial_bp//resolution,size=num, dtype='l')
    # to not include repeats use choice
    
    p_array=[]
    APA_array=[]
    for a,b in loops_chrom[[1,4]].values:
        #generate list of random distances in bins for each loop
        pos=np.random.choice(range(distance_initial_bp//resolution,distance_bp//resolution), num, replace=False)
        neg=np.random.choice(range(-distance_bp//resolution,-distance_initial_bp//resolution), num, replace=False)
        rnd_bins=np.concatenate((pos,neg))
        rnd_APA=[]
        APA=APA_analysis(int(a),int(b),mtx,return_PA=PA)
        #print('ab: ',[a,b])
        #print('rnd: ',rnd_bins)
        for el in rnd_bins:
            a_r=int(a)+int(el)
            b_r=int(b)+int(el)
            #test approach!!!! should check in future
            if a_r+2>=s or b_r+2>=s or a_r+1>=s or b_r+1>=s:
                r=np.random.choice(range(5,20),1,replace=False)
                a_r=int(a)-int(r)
                b_r=int(b)-int(r)
            if a_r-2<=1 or b_r-2<=1 or a_r-1<=1 or b_r-1<=1:
                r=np.random.choice(range(5,20),1,replace=False)
                a_r=int(a)+int(r)
                b_r=int(b)+int(r)
            #print([a_r,b_r])
            rnd_APA.append(APA_analysis(a_r,b_r,mtx,return_PA=PA))
        #p_value=scipy.stats.ttest_1samp(rnd_APA,APA)[1]
        #p_values.append(p_value)
        #p=np.percentile(rnd_APA,q=percentile)
        p=np.quantile(rnd_APA,q=percentile)
        p_array.append(p)
        APA_array.append(APA)
    return(APA_array,p_array)

