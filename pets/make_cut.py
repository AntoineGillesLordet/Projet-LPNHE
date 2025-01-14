import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob,os,re,sys
import sncosmo
from astropy.table import Table,vstack
from scipy.optimize import leastsq,minimize
import math
from lemaitre import bandpasses
from joblib import Parallel, delayed
from scipy.interpolate import interp1d
from scipy.signal import argrelmin
import saltworks
import scipy.stats

from lemaitre import bandpasses
filterlib = bandpasses.get_filterlib()

import warnings
from iminuit.warnings import IMinuitWarning

warnings.filterwarnings("ignore", category=IMinuitWarning)

from tqdm.auto import tqdm

m0file='nacl_m0_test.dat'
m1file='nacl_m1_test.dat'
clfile='nacl_color_law_test.dat'
modelpath='../data/SALT_snf/'

def interpolated_intercepts(x, y1, y):
	"""Find the intercepts of two curves, given by the same x data"""
	y2=np.zeros(np.size(x))+y
	def intercept(point1, point2, point3, point4):
		"""find the intersection between two lines
		the first line is defined by the line between point1 and point2
		the first line is defined by the line between point3 and point4
		each point is an (x,y) tuple.

		So, for example, you can find the intersection between
		intercept((0,0), (1,1), (0,1), (1,0)) = (0.5, 0.5)

		Returns: the intercept, in (x,y) format
		"""    

		def line(p1, p2):
			A = (p1[1] - p2[1])
			B = (p2[0] - p1[0])
			C = (p1[0]*p2[1] - p2[0]*p1[1])
			return A, B, -C

		def intersection(L1, L2):
			D  = L1[0] * L2[1] - L1[1] * L2[0]
			Dx = L1[2] * L2[1] - L1[1] * L2[2]
	
			x = Dx / D
			return x

		L1 = line([point1[0],point1[1]], [point2[0],point2[1]])
		L2 = line([point3[0],point3[1]], [point4[0],point4[1]])
	
		R = intersection(L1, L2)

		return R

	idxs = np.argwhere(np.diff(np.sign(y1 - y2)) != 0)

	xcs=[intercept((x[idx], y1[idx]),((x[idx+1], y1[idx+1])), ((x[idx], y2[idx])), ((x[idx+1], y2[idx+1])))[0] for i,idx in enumerate(idxs)]
	return np.array(xcs)


def Tmax_chi2(name,lc_ztf,data_ztf,x1_lim=4,c_lim=1,showfig=False,savefig=False,output_fig=None):

	df_Tmax_1SN=pd.DataFrame(columns=['name','Tsncosmo','eTsncosmo','Tchi2','eTchi2'])
	#Select data for specific SN
	lc_sn=lc_ztf[lc_ztf.name==name]
	#Remove i band for fitting
	#lc_sn=lc_sn[lc_sn['band']!='ztf::I'] 
	#Redshift and MW extinction
	mwebv=data_ztf[data_ztf.name==name].mwebv.values[0]
	zsn=data_ztf[data_ztf.name==name].zhel.values[0]
	#Transform to Table for sncosmo
	lc_sncosmo=Table.from_pandas(lc_sn[['mjd','band','flux','fluxerr','zp','magsys','valid']])
	lc_sncosmo=lc_sncosmo[lc_sncosmo['valid']==1]		
	SNtype=data_ztf[data_ztf.name==name].classification.values[0]
	# create a model
	source = sncosmo.SALT2Source(modeldir=modelpath,m0file=m0file, m1file=m1file, clfile=clfile)
	dust = sncosmo.CCM89Dust()
	model= sncosmo.Model(source=source,effects=[dust],effect_names=['mw'],effect_frames=['obs'])
	
	model.set(z=zsn,mwebv=mwebv,mwr_v=3.1)  # set the model's redshift and MW
	#First fit to get a starting position for MCMC

	if np.size(lc_sncosmo)>0:
		try:
		
			res, mod = sncosmo.fit_lc(lc_sncosmo, model,['t0', 'x0', 'x1', 'c'],bounds={'x0':(-0.1,10),'x1':(-5, 5),'c':(-3, 3)},phase_range=None,modelcov=False)
			sncosmo.plot_lc(lc_sncosmo, model=mod, errors=res.errors)
			if savefig:
			
				plt.savefig(output_fig+'%s_sncosmo.png'%name)

			if showfig:
				plt.show()
				plt.close()
			plt.close()
			
			Tsncosmo=res.parameters[1]
			eTsncosmo=res.errors['t0']
			c_sncosmo=res.parameters[4]
			x1_sncosmo=res.parameters[3]
			ec_sncosmo=res.errors['c']
			ex1_sncosmo=res.errors['x1']
			chi2sncosmo=res.chisq	

		except:
			Tsncosmo=999.9
			eTsncosmo=999.9		
			c_sncosmo=999.9
			x1_sncosmo=999.9
			ec_sncosmo=999.9
			ex1_sncosmo=999.9	
			chi2sncosmo=999.9			

	else:
										
		Tsncosmo=999.9
		eTsncosmo=999.9		
		c_sncosmo=999.9
		x1_sncosmo=999.9
		ec_sncosmo=999.9
		ex1_sncosmo=999.9	
		chi2sncosmo=999.9
		
		Tsncosmo_fit=999.9
		eTsncosmo_fit=999.9		
		c_sncosmo_fit=999.9
		x1_sncosmo_fit=999.9
		ec_sncosmo_fit=999.9
		ex1_sncosmo_fit=999.9
		x0_sncosmo_fit=999.9
		ex0_sncosmo_fit=999.9	
					
	if os.path.isfile('Tgrid/%s.dat' %name):
		grid=np.loadtxt('Tgrid/%s.dat' %name).transpose()	
	
		if (np.size(grid)>10):

			Tmax,x0_grid,ex0_grid,x1_grid,ex1_grid,c_grid,ec_grid,chi2_grid=grid[0],grid[1],grid[2],grid[3],grid[4],grid[5],grid[6],grid[-2]
            
			ndof=grid[-1,0]
			func_chi2=np.vectorize(interp1d(Tmax,chi2_grid))
			
			#Look for all the minimum
			min_local=np.unique(argrelmin(chi2_grid, mode='wrap')[0])
			Tmin_tot,chi2min_tot,x0_tot,ex0_tot,x1_tot,ex1_tot,c_tot,ec_tot=Tmax[min_local],chi2_grid[min_local],x0_grid[min_local],ex0_grid[min_local],x1_grid[min_local],ex1_grid[min_local],c_grid[min_local],ec_grid[min_local]
			
			#Remove minimum where X1 and c out limit
			Tmin_all=Tmin_tot[(np.abs(x1_grid[min_local])<x1_lim) & (np.abs(c_grid[min_local])<c_lim)]
			chi2min_all=chi2min_tot[(np.abs(x1_grid[min_local])<x1_lim) & (np.abs(c_grid[min_local])<c_lim)]
			x0_all=x0_tot[(np.abs(x1_grid[min_local])<x1_lim) & (np.abs(c_grid[min_local])<c_lim)]
			ex0_all=ex0_tot[(np.abs(x1_grid[min_local])<x1_lim) & (np.abs(c_grid[min_local])<c_lim)]
			x1_all=x1_tot[(np.abs(x1_grid[min_local])<x1_lim) & (np.abs(c_grid[min_local])<c_lim)]
			ex1_all=ex1_tot[(np.abs(x1_grid[min_local])<x1_lim) & (np.abs(c_grid[min_local])<c_lim)]
			c_all=c_tot[(np.abs(x1_grid[min_local])<x1_lim) & (np.abs(c_grid[min_local])<c_lim)]
			ec_all=ec_tot[(np.abs(x1_grid[min_local])<x1_lim) & (np.abs(c_grid[min_local])<c_lim)]

			n_min=np.size(Tmin_tot)
			if np.size(Tmin_all)>0:
				ind_Tmin=np.where(chi2min_all==min(chi2min_all))[0][0]
					
				Tchi2=Tmin_all[ind_Tmin]
				chi2min=chi2min_all[ind_Tmin]
				x1_chi2=x1_all[ind_Tmin]
				x0_chi2=x0_all[ind_Tmin]					
				c_chi2=c_all[ind_Tmin]
				ex1_chi2=ex1_all[ind_Tmin]
				ec_chi2=ec_all[ind_Tmin]				
				ex0_chi2=ex0_all[ind_Tmin]					
				#Confidence 	   68.27%	   95.45%	   99.73%	   99.99%	  100.00%
				#p-value    	 0.31731	 0.04550	 0.00270	 0.00006	 0.00000
				#sigma(k=1) 	    1.00	    2.00	    3.00	    4.00	    5.00
				#chi2(d=1) 	    1.00	    4.00	    9.00	   16.00	   25.00

				n_min_8sig=np.size(chi2min_all[chi2min_all<chi2min+64])
			else:
				Tchi2=999.9	
				chi2min=999.9
				n_min_8sig=999.9
				x1_chi2=999.9
				c_chi2=999.9
				ex1_chi2=999.9
				ec_chi2=999.9
				x0_chi2=999.9
				ex0_chi2=999.9				

			#Calcul error max and min for each mininim
			err_Tmax=interpolated_intercepts(Tmax,func_chi2(Tmax),chi2min+1)

			#Calcul 3 sig error max and min for each mininim
			err_Tmax3sig=interpolated_intercepts(Tmax,func_chi2(Tmax),chi2min+9)

			try:
				max_fig=min(np.array(err_Tmax)[err_Tmax>Tchi2])	
				min_fig=max(np.array(err_Tmax)[err_Tmax<Tchi2])
				eTchi2=(np.sqrt((max_fig-Tchi2)**2+(Tchi2-min_fig)**2)/np.sqrt(2))
				eTchi2_min=Tchi2-min_fig
				eTchi2_max=max_fig-Tchi2
			except:
				eTchi2=999.9
				eTchi2_min=999.9
				eTchi2_max=999.9
				max_fig=999.9	
				min_fig=999.9	

			try:
				max_fig3sig=min(np.array(err_Tmax3sig)[err_Tmax3sig>Tchi2])	
				min_fig3sig=max(np.array(err_Tmax3sig)[err_Tmax3sig<Tchi2])
				eTchi2_3sig=(np.sqrt((max_fig3sig-Tchi2)**2+(Tchi2-min_fig3sig)**2)/np.sqrt(2))
				eTchi2_3sig_min=Tchi2-min_fig3sig
				eTchi2_3sig_max=max_fig3sig-Tchi2
			except:
				eTchi2_3sig=999.9
				eTchi2_3sig_min=999.9
				eTchi2_3sig_max=999.9
				max_fig3sig=999.9	
				min_fig3sig=999.9

			
			if (eTchi2_min!=999.9) and (eTchi2_max!=999.9) and (eTchi2_3sig_min!=999.9) and (eTchi2_3sig_max!=999.9):


				#Refit using Tchi2
				source = sncosmo.SALT2Source(modeldir=modelpath,m0file=m0file, m1file=m1file, clfile=clfile)
				dust = sncosmo.CCM89Dust()
				model_fit= sncosmo.Model(source=source,effects=[dust],effect_names=['mw'],effect_frames=['obs'])
			
				model_fit.set(z=zsn,mwebv=mwebv,mwr_v=3.1,t0=Tchi2)  # set the model's redshift and MW
				#First fit to get a starting position for MCMC

				try:
					res_fit, mod_fit = sncosmo.fit_lc(lc_sncosmo, model_fit,['t0', 'x0', 'x1', 'c'],phase_range=None,modelcov=False,guess_t0=False)
					chi2sncosmo_fit=res_fit.chisq


					Tsncosmo_fit,x0_sncosmo_fit,x1_sncosmo_fit,c_sncosmo_fit=res_fit.parameters[1:5]
					eTsncosmo_fit,ex0_sncosmo_fit,ex1_sncosmo_fit,ec_sncosmo_fit=res_fit.errors['t0'],res_fit.errors['x0'],res_fit.errors['x1'],res_fit.errors['c']
				except:
					Tsncosmo_fit,x0_sncosmo_fit,x1_sncosmo_fit,c_sncosmo_fit=999.9,999.9,999.9,999.9								
					eTsncosmo_fit,ex0_sncosmo_fit,ex1_sncosmo_fit,ec_sncosmo_fit=999.9,999.9,999.9,999.9								
					chi2sncosmo_fit=999.9					

				if showfig:
				
					#Figures
					fig, ax1 = plt.subplots(figsize=(8,6), facecolor='w', edgecolor='k')
					ax1.plot(Tmax,chi2_grid,color='black',marker='s',alpha=0.6,ms=3,linestyle=None)


					ax1.set_ylabel('Chisq',fontsize=20,fontweight='bold')
					ax1.set_xlabel('T0',fontsize=20,fontweight='bold')
						

					ax1.axhline(y=new_chi2,color='black')
					ax1.axhline(y=new_chi2+8,color='grey')
					ax1.axvline(x=Tsncosmo,color='r',label='SN cosmo:%0.1f+/-%0.3f'%(Tsncosmo,eTsncosmo))
					if Tchi2!=999.9:
						ax1.axvline(x=Tchi2,color='b',alpha=0.6)
						ax1.axvspan(min_fig,max_fig, alpha=0.3, color='blue',label='%0.1f+%0.3f-%0.3f'%(Tchi2,max_fig-Tchi2,Tchi2-min_fig),zorder=2)
						ax1.axvspan(min_fig3sig,max_fig3sig, alpha=0.1, color='blue',zorder=2)		
					#ax1.set_xlim([Tchi2-20, Tchi2+20])
					#ax1.set_ylim([chi2min*0.95, chi2min*1.8])  
					

					ax1.tick_params(axis='both', which='major', labelsize=20,direction='in',right='on',top='on')
					ax1.text(0.1, 0.2, '%s'%SNtype, horizontalalignment='center',verticalalignment='center', transform=ax1.transAxes, size=18, fontdict=None)			

					ax1.plot(Tmax[(np.abs(x1_grid)<x1_lim) & (np.abs(c_grid)<c_lim)],chi2_grid[(np.abs(x1_grid)<x1_lim) & (np.abs(c_grid)<c_lim)],color='b',marker='s',alpha=0.6,ms=3,linestyle=None,label='|x1|<%s;|c|<%s'%(x1_lim,c_lim))


					ax1.legend(loc=0,markerscale=1.0,prop={'size':14},ncol=1) 
					
							
					if savefig:
						plt.savefig(output_fig+'%s_Tmaxgrid.png'%name,bbox_inches='tight')
					plt.show()
					plt.close()

			else:

				eTchi2=999.9
				eTchi2_3sig=999.9
				eTchi2_min=999.9
				eTchi2_max=999.9
				eTchi2_3sig_min=999.9
				eTchi2_3sig_max=999.9

				Tchi2=999.9
				n_min=999.9
				n_min_8sig=999.9
				c_chi2=999.9
				x1_chi2=999.9
				ec_chi2=999.9
				ex1_chi2=999.9
				x0_chi2=999.9
				ex0_chi2=999.9				
				
				Tsncosmo_fit=999.9
				eTsncosmo_fit=999.9	
				c_sncosmo_fit=999.9
				x1_sncosmo_fit=999.9
				ec_sncosmo_fit=999.9
				ex1_sncosmo_fit=999.9
				x0_sncosmo_fit=999.9
				ex0_sncosmo_fit=999.9

				Tsncosmo=999.9
				esncosmo=999.9
				chi2min=999.9
				chi2sncosmo=999.9		
				chi2sncosmo_fit=999.9	

				
		else:

			eTchi2=999.9
			eTchi2_3sig=999.9
			eTchi2_min=999.9
			eTchi2_max=999.9
			eTchi2_3sig_min=999.9
			eTchi2_3sig_max=999.9

			Tchi2=999.9
			n_min=999.9
			n_min_8sig=999.9
			c_chi2=999.9
			x1_chi2=999.9
			ec_chi2=999.9
			ex1_chi2=999.9
			x0_chi2=999.9
			ex0_chi2=999.9				
			
			Tsncosmo_fit=999.9
			eTsncosmo_fit=999.9	
			c_sncosmo_fit=999.9
			x1_sncosmo_fit=999.9
			ec_sncosmo_fit=999.9
			ex1_sncosmo_fit=999.9
			x0_sncosmo_fit=999.9
			ex0_sncosmo_fit=999.9

			Tsncosmo=999.9
			esncosmo=999.9
			chi2min=999.9
			chi2sncosmo=999.9		
			chi2sncosmo_fit=999.9							
			
	
	else:
		eTchi2=999.9
		eTchi2_3sig=999.9
		eTchi2_min=999.9
		eTchi2_max=999.9
		eTchi2_3sig_min=999.9
		eTchi2_3sig_max=999.9
		Tchi2=999.9
		n_min=999.9
		n_min_8sig=999.9
		c_chi2=999.9
		x1_chi2=999.9
		ec_chi2=999.9
		ex1_chi2=999.9
		
		x0_chi2=999.9
		ex0_chi2=999.9				
		
		Tsncosmo_fit=999.9
		eTsncosmo_fit=999.9	
		c_sncosmo_fit=999.9
		x1_sncosmo_fit=999.9
		ec_sncosmo_fit=999.9
		ex1_sncosmo_fit=999.9
		x0_sncosmo_fit=999.9
		ex0_sncosmo_fit=999.9
		chi2min=999.9	
		chi2sncosmo=999.9		
		chi2sncosmo_fit=999.9				

	df_Tmax_1SN['name']=np.array([name])
	df_Tmax_1SN['zhel']=np.array([zsn])
	df_Tmax_1SN['mebv']=np.array([mwebv])
	df_Tmax_1SN['Tsncosmo']=np.array([Tsncosmo])
	df_Tmax_1SN['eTsncosmo']=np.array([eTsncosmo])
	df_Tmax_1SN['Tchi2']=np.array([Tchi2])
	df_Tmax_1SN['eTchi2']=np.array([eTchi2])
	df_Tmax_1SN['eTchi2_min']=np.array([eTchi2_min])
	df_Tmax_1SN['eTchi2_max']=np.array([eTchi2_max])
	df_Tmax_1SN['eTchi_3sig_min']=np.array([eTchi2_3sig_min])
	df_Tmax_1SN['eTchi_3sig_max']=np.array([eTchi2_3sig_max])

	df_Tmax_1SN['x0_chi2']=np.array([x0_chi2])
	df_Tmax_1SN['ex0_chi2']=np.array([ex0_chi2])
	df_Tmax_1SN['x1_chi2']=np.array([x1_chi2])
	df_Tmax_1SN['ex1_chi2']=np.array([ex1_chi2])
	df_Tmax_1SN['c_chi2']=np.array([c_chi2])
	df_Tmax_1SN['ec_chi2']=np.array([ec_chi2])
	df_Tmax_1SN['n_min']=np.array([n_min])
	df_Tmax_1SN['n_min_8sig']=np.array([n_min_8sig])

	df_Tmax_1SN['Tsncosmo_fit']=np.array([Tsncosmo_fit])
	df_Tmax_1SN['eTsncosmo_fit']=np.array([eTsncosmo_fit])
	df_Tmax_1SN['x0_sncosmo']=np.array([x0_sncosmo_fit])
	df_Tmax_1SN['ex0_sncosmo']=np.array([ex0_sncosmo_fit])
	df_Tmax_1SN['x1_sncosmo']=np.array([x1_sncosmo_fit])
	df_Tmax_1SN['ex1_sncosmo']=np.array([ex1_sncosmo_fit])
	df_Tmax_1SN['c_sncosmo']=np.array([c_sncosmo_fit])
	df_Tmax_1SN['ec_sncosmo']=np.array([ec_sncosmo_fit])

	df_Tmax_1SN['X2min']=np.array([chi2min])
	df_Tmax_1SN['x2sncosmo']=np.array([chi2sncosmo])
	df_Tmax_1SN['x2sncosmo_fit']=np.array([chi2sncosmo_fit])
	return df_Tmax_1SN



data_ztf=pd.read_csv('mock_sne.csv')
lc_ztf=pd.read_csv('mock_lc.csv')

data_ztf=data_ztf.sort_values(by=['name'])

name=data_ztf.name.values

tmax_file=input('Do you want to create Tmax_mock.csv ? y or n: '	)

if tmax_file=='y':

	df_Tmax_tot=pd.DataFrame(columns=['name','Tsncosmo','eTsncosmo','Tchi2','eTchi2'])
	for nn in tqdm(name, desc='Finding chi2 minima'):
		df_Tmax_1SN=Tmax_chi2(nn,lc_ztf,data_ztf,x1_lim=4,c_lim=1,showfig=False,savefig=False,output_fig=None)
		df_Tmax_tot=pd.concat([df_Tmax_tot,df_Tmax_1SN])
		plt.close('all')
	df_Tmax_tot.to_csv('Tmax_mock.csv',index=False) 	

	
	
df_Tmax=pd.read_csv('Tmax_mock.csv')	

data_ztf = pd.merge(data_ztf[['name','zhel','zcmb','ra','dec','mwebv','classification','sn','comments']], df_Tmax[['name','Tchi2','x0_chi2','x1_chi2','c_chi2']],  how='right', left_on=['name'], right_on = ['name'])			
	
df_Tmax['flag']=0			
	
eTlim=1
symlim=0.3
x1lim=4
clim=2
cutx1=False
cutc=False	

#					flag 	cut criteria 				purpose
#					1 	sncosmo converged 			have data, have found a minimum
#					2 	eTchi2<1=eTlim 			Tmax well defined
#					4 	abs(eTmax-eTmin) 3sig<0.3=symlim 	minimum should be symetric
#					8 	Only 1 min at 3sig			Only one clear minimum
#					16 	abs(X1_chi2)<4=x1lim	 		cut in X1
#					32 	abs(c_chi2)<2=clim	 		cut in color


print('Cut\t\t\t\t\tDiscarded\tRemaining')
print('Tot\t\t\t\t\t-\t\t%s'%df_Tmax.shape[0])	

df_Tmax.loc[~(df_Tmax != 999.9).all(axis=1),'flag']=1


nsn=df_Tmax[df_Tmax.flag==0].shape[0]
print('sncosmo converged\t\t\t%s\t\t%s'%(df_Tmax.shape[0]-nsn,nsn))

# Make the cut

#First, select only those with an error smaller than 1d

df_Tmax.loc[df_Tmax['eTchi2']>eTlim,'flag']+=2
nsn=df_Tmax[df_Tmax.flag==2].shape[0]
print('eTmax<%s\t\t\t\t\t%s\t\t%s'%(eTlim,nsn,df_Tmax[df_Tmax.flag==0].shape[0]))


#Symetric error at 3 sigma
df_Tmax.loc[np.abs(df_Tmax.eTchi_3sig_max-df_Tmax.eTchi_3sig_min)>symlim,'flag']+=4
nsn=df_Tmax[df_Tmax.flag==4].shape[0]
print('abs(eTmax-eTmin) 5sig<%s\t\t%s\t\t%s'%(symlim,nsn,df_Tmax[df_Tmax.flag==0].shape[0]))


#Check if there is another close minimum (less than 3 sigma)
df_Tmax.loc[df_Tmax.n_min_8sig!=1,'flag']+=8
nsn=df_Tmax[df_Tmax.flag==8].shape[0]
print('Only 1 min at 8sig\t\t\t%s\t\t%s'%(nsn,df_Tmax[df_Tmax.flag==0].shape[0]))


#Check X1

df_Tmax.loc[df_Tmax.x1_chi2>x1lim,'flag']+=16
nsn=df_Tmax[df_Tmax.flag==16].shape[0]

print('abs(X1_chi2)<%s\t\t\t\t%s\t\t%s'%(x1lim,nsn,df_Tmax[df_Tmax.flag==0].shape[0]))

#Check c
df_Tmax.loc[df_Tmax.c_chi2>clim,'flag']+=32
nsn=df_Tmax[df_Tmax.flag==32].shape[0]
print('abs(col_chi2)<%s\t\t\t\t%s\t\t%s'%(clim,nsn,df_Tmax[df_Tmax.flag==0].shape[0]))

df_Tmax.to_csv('Tmax_mock.csv', index=False)

#cut
if (cutx1==True) and (cutc==True):
	df_Tmax=df_Tmax[df_Tmax.flag==0]
elif (cutx1==False) and (cutc==False):
	df_Tmax=df_Tmax[(df_Tmax.flag==0) | (df_Tmax.flag==16) | (df_Tmax.flag==32)]
elif (cutx1==True) and (cutc==False):
	df_Tmax=df_Tmax[(df_Tmax.flag==0) | (df_Tmax.flag==32)]
elif (cutx1==False) and (cutc==True):
	df_Tmax=df_Tmax[(df_Tmax.flag==0) | (df_Tmax.flag==16) ]




data_ztf=data_ztf.rename(columns={'Tchi2': 'tmax', 'x0_chi2': 'x0', 'x1_chi2': 'x1', 'c_chi2': 'c'})                         
data_ztf['survey']='ZTF'
data_ztf['valid']=1

data_ztf['valid'] = data_ztf.apply(lambda row: 0 if row['name'] not in df_Tmax['name'].values else row['valid'], axis=1)


lc_ztf['valid'] = lc_ztf.apply(lambda row: 0 if row['name'] not in df_Tmax['name'].values else row['valid'], axis=1)



data_ztf.to_csv('Results/mock_sne.csv', index=False)
lc_ztf.to_csv('Results/mock_lc.csv', index=False)



