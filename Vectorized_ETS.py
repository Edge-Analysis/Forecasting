import pandas as pd
import numpy as np
import multiprocessing as mp


class fcst:
    def __init__(self,hist,params=None, fcst_periods=18, season_periods = 12):
        self.hist = hist
        self.hist_periods = len(hist)
        self.params = params
        self.fcst_periods = fcst_periods
        self.season_periods = season_periods
        
    def init_param(self):
        
        param_df = self.hist[:5].copy().reset_index().drop('Period',axis=1)
        param_df.rename({0:'Alpha',1:'Beta',2:'Gamma',3:'Phi',4:'Start_per'}, axis='index', inplace=True)
        
        param_df[:1] = 0.5
        param_df[1:2] = 0.5
        param_df[2:3] = 0.5
        param_df[3:4] = 1.0
        param_df[4:5] = 0
        
        season_add = self.hist[-self.season_periods:].copy().reset_index().drop('Period',axis=1)
        season_add *= 0
        season_mul = season_add.copy()
        seasons = (self.hist.count()/self.season_periods).astype(int)
        
        for i in range(len(self.hist.columns)):
            first_period = self.hist.iloc[:,i].to_frame().dropna().iloc[0].name -1
            param_df.iloc[4,i] = first_period
            season_avg = 0
            for s in range(seasons[i]):
                season_avg = self.hist.iloc[self.season_periods*s + first_period:self.season_periods*(s+1) + first_period,i].sum() / self.season_periods
                for m in range(self.season_periods):
                    season_add.iloc[m, i] += self.hist.iloc[self.season_periods*s + first_period + m,i] - season_avg
                    season_mul.iloc[m, i] += self.hist.iloc[self.season_periods*s + first_period + m,i] / season_avg
                    
        season_add /= seasons
        season_mul /= seasons
        
        return param_df, season_add, season_mul
    
    def init_dfs(self,trend=False,season=False,season_init=False):
        level_df = self.hist.copy() * 0
        trend_df = level_df.copy()
        season_df = level_df.copy()
        
        level_df += self.params.iloc[0]
        for i in range(len(level_df.columns)):
            level_df.iloc[int(self.params.iloc[4,i]),i] = 1
            
        level_df.iloc[0] = self.hist.iloc[0]
        level_df = level_df.fillna(0)
        fcst_df = level_df.copy()
        
        trend_init = {}
        if trend == True:
            trend_df += self.params.iloc[1]
            for i in range(len(trend_df.columns)):
                trend_df.iloc[int(self.params.iloc[4,i]),i] = float('nan')
                trend_init[trend_df.columns[i]] = self.hist.iloc[int(self.params.iloc[4,i])+1,i] - self.hist.iloc[int(self.params.iloc[4,i]),i]
            
            trend_df.iloc[0] = self.hist.iloc[1] - self.hist.iloc[0]
            trend_df = trend_df.fillna(0)
            
        if season == True:
            for i in range(len(season_df.columns)):
                start = int(self.params.iloc[4,i])
                for j in range(0,self.season_periods):
                    if start - self.season_periods + j >=0:
                        season_df.iloc[start - self.season_periods + j, i] = season_init.iloc[j,i]
                    else:
                        season_df.iloc[start + j, i] = season_init.iloc[j,i]
            season_df = season_df.fillna(0)
            
            
        return fcst_df, level_df, [trend_df,trend_init], season_df
    
    def exsm_NN(self):
        fcst_df, level_df, trend, season_df = self.init_dfs()
        
        for i in range(1,self.hist_periods):
            level_df.iloc[i] = (level_df.iloc[i] * self.hist.iloc[i].fillna(0)) + ((1 - level_df.iloc[i]) * level_df.iloc[i-1])
            
        fin_level = np.array(level_df.iloc[-1])
        for i in range(1,self.hist_periods+self.fcst_periods + 1): 
            fcst_df.loc[i] = fin_level
        
        return fcst_df
    
    def exsm_NA(self, season_add):
        fcst_df, level_df, trend, season_df = self.init_dfs(season=True,season_init=season_add)
        
        for i in range(1,self.hist_periods):
            if i < self.season_periods:
                level_df.iloc[i] = (level_df.iloc[i] * (self.hist.iloc[i].fillna(0)-season_df.iloc[i])) + ((1 - level_df.iloc[i]) * level_df.iloc[i-1])
            else:
                level_df.iloc[i] = (level_df.iloc[i] * (self.hist.iloc[i].fillna(0)-season_df.iloc[i-self.season_periods])) + ((1 - level_df.iloc[i]) * level_df.iloc[i-1])
            
            if i >= self.season_periods:
                season_df.iloc[i] += self.params.iloc[2] * (self.hist.iloc[i].fillna(0)-level_df.iloc[i]) + ((1-self.params.iloc[2])*season_df.iloc[i-self.season_periods])
        fin_level = np.array(level_df.iloc[-1])
        
        for i in range(self.hist_periods-self.season_periods,0, -1):
            season_df.loc[i] = season_df.loc[i + self.season_periods]
            fcst_df.loc[i] = season_df.loc[i] + fin_level
        for i in range(self.hist_periods-self.season_periods,self.hist_periods+self.fcst_periods+1):
            season_df.loc[i] = season_df.loc[i - self.season_periods]
            fcst_df.loc[i] = season_df.loc[i] + fin_level
            
        fcst_df[fcst_df < 0] = 0
        
        return fcst_df
    
    def exsm_NM(self, season_mul):
        fcst_df, level_df, trend, season_df = self.init_dfs(season=True,season_init=season_mul)
        
        for i in range(1,self.hist_periods):
            if i < self.season_periods:
                level_df.iloc[i] = (level_df.iloc[i] * (self.hist.iloc[i].fillna(0)/season_df.iloc[i])) + ((1 - level_df.iloc[i]) * level_df.iloc[i-1])
            else:
                level_df.iloc[i] = (level_df.iloc[i] * (self.hist.iloc[i].fillna(0)/season_df.iloc[i-self.season_periods])) + ((1 - level_df.iloc[i]) * level_df.iloc[i-1])
                
            level_df.iloc[i] = level_df.iloc[i].fillna(0)
            if i >= self.season_periods:
                season_df.iloc[i] += self.params.iloc[2] * (self.hist.iloc[i].fillna(0)/level_df.iloc[i]).fillna(0) + ((1-self.params.iloc[2])*season_df.iloc[i-self.season_periods])

        fin_level = np.array(level_df.iloc[-1])
        
        for i in range(self.hist_periods-self.season_periods,0, -1):
            season_df.loc[i] = season_df.loc[i + self.season_periods]
            fcst_df.loc[i] = season_df.loc[i] * fin_level
        for i in range(self.hist_periods-self.season_periods,self.hist_periods+self.fcst_periods+1):
            season_df.loc[i] = season_df.loc[i - self.season_periods]
            fcst_df.loc[i] = season_df.loc[i] * fin_level
            
        fcst_df[fcst_df < 0] = 0
        
        return fcst_df
    
    def exsm_LN(self):
        fcst_df, level_df, trend, season_df = self.init_dfs(trend=True)
        
        trend_df = trend[0]
        trend_init = trend[1]
        
        for i in range(1,self.hist_periods):
            level_df.iloc[i] = (level_df.iloc[i] * self.hist.iloc[i].fillna(0)) + ((1 - level_df.iloc[i]) * (level_df.iloc[i-1] + trend_df.iloc[i-1]))
            trend_df.iloc[i-1] = trend_df.iloc[i-1].fillna(trend_init)
            trend_df.iloc[i] = (trend_df.iloc[i] * (level_df.iloc[i] - level_df.iloc[i-1])) + ((1 - trend_df.iloc[i]) * trend_df.iloc[i-1])

        fin_level = np.array(level_df.iloc[-1])
        fin_trend = np.array(trend_df.iloc[-1])
        
        fcst_df.loc[self.hist_periods] = fin_level
        power = 0
        for i in range(self.hist_periods-1,0, -1):
            power += 1
            fcst_df.loc[i] = fcst_df.loc[i+1] - fin_trend * (self.params.iloc[3] ** power)
        
        power = 0
        for i in range(self.hist_periods + 1, self.hist_periods+self.fcst_periods + 1):
            power += 1
            fcst_df.loc[i] = fcst_df.loc[i-1] + fin_trend * (self.params.iloc[3] ** power)
            
        fcst_df[fcst_df < 0] = 0
        
        return fcst_df
    
    def exsm_LA(self, season_add):
        fcst_df, level_df, trend, season_df = self.init_dfs(trend=True,season=True,season_init=season_add)
        
        trend_df = trend[0]
        trend_init = trend[1]
        
        for i in range(1,self.hist_periods):
            if i < self.season_periods:
                level_df.iloc[i] = (level_df.iloc[i] * (self.hist.iloc[i].fillna(0)-season_df.iloc[i])) + ((1 - level_df.iloc[i]) * (level_df.iloc[i-1] + trend_df.iloc[i-1]))
            else:
                level_df.iloc[i] = (level_df.iloc[i] * (self.hist.iloc[i].fillna(0)-season_df.iloc[i-self.season_periods])) + ((1 - level_df.iloc[i]) * (level_df.iloc[i-1] + trend_df.iloc[i-1]))
            
            trend_df.iloc[i-1] = trend_df.iloc[i-1].fillna(trend_init)
            trend_df.iloc[i] = (trend_df.iloc[i] * (level_df.iloc[i] - level_df.iloc[i-1])) + ((1 - trend_df.iloc[i]) * trend_df.iloc[i-1])
            
            if i >= self.season_periods:
                season_df.iloc[i] += self.params.iloc[2] * (self.hist.iloc[i].fillna(0)-level_df.iloc[i]) + ((1-self.params.iloc[2])*season_df.iloc[i-self.season_periods])

        fin_level = np.array(level_df.iloc[-1])
        fin_trend = np.array(trend_df.iloc[-1])
        fcst_df.loc[self.hist_periods] = fin_level + season_df.iloc[-1]
        power = 0
        for i in range(self.hist_periods-1,0, -1):
            power += 1
            if power > self.season_periods:
                season_df.loc[i] = season_df.loc[i + self.season_periods]
            fcst_df.loc[i] = fcst_df.loc[i+1] - (fin_trend * (self.params.iloc[3] ** power)) + season_df.loc[i]
        
        power = 0
        for i in range(self.hist_periods + 1, self.hist_periods+self.fcst_periods + 1):
            power += 1
            season_df.loc[i] = season_df.loc[i - self.season_periods]
            fcst_df.loc[i] = fcst_df.loc[i-1] + fin_trend * (self.params.iloc[3] ** power) + season_df.loc[i]
        
        fcst_df[fcst_df < 0] = 0
            
        return fcst_df
    
    def exsm_LM(self, season_mul):
        fcst_df, level_df, trend, season_df = self.init_dfs(trend=True,season=True,season_init=season_mul)
        
        trend_df = trend[0]
        trend_init = trend[1]
        
        for i in range(1,self.hist_periods):
            
            if i < self.season_periods:
                level_df.iloc[i] = (level_df.iloc[i] * (self.hist.iloc[i].fillna(0)/season_df.iloc[i])) + ((1 - level_df.iloc[i]) * (level_df.iloc[i-1] + trend_df.iloc[i-1]))
            else:
                level_df.iloc[i] = (level_df.iloc[i] * (self.hist.iloc[i].fillna(0)/season_df.iloc[i-self.season_periods])) + ((1 - level_df.iloc[i]) * (level_df.iloc[i-1] + trend_df.iloc[i-1]))
            
            level_df.iloc[i] = level_df.iloc[i].fillna(0)
            trend_df.iloc[i-1] = trend_df.iloc[i-1].fillna(trend_init)
            trend_df.iloc[i] = (trend_df.iloc[i] * (level_df.iloc[i] - level_df.iloc[i-1])) + ((1 - trend_df.iloc[i]) * trend_df.iloc[i-1])
            
            if i >= self.season_periods:
                season_df.iloc[i] += self.params.iloc[2] * (self.hist.iloc[i].fillna(0)/level_df.iloc[i]).fillna(0) + ((1-self.params.iloc[2])*season_df.iloc[i-self.season_periods])
        
        fin_level = np.array(level_df.iloc[-1])
        fin_trend = np.array(trend_df.iloc[-1])
        fcst_df.loc[self.hist_periods] = fin_level * season_df.iloc[-1]
        
        power = 0
        for i in range(self.hist_periods-1,0, -1):
            power += 1
            if power > self.season_periods:
                season_df.loc[i] = season_df.loc[i + self.season_periods]
            fcst_df.loc[i] = (fcst_df.loc[i+1] - (fin_trend * (self.params.iloc[3] ** power))) * season_df.loc[i]
        
        power = 0
        for i in range(self.hist_periods + 1, self.hist_periods+self.fcst_periods + 1):
            power += 1
            season_df.loc[i] = season_df.loc[i - self.season_periods]
            fcst_df.loc[i] = (fcst_df.loc[i-1] + fin_trend * (self.params.iloc[3] ** power)) * season_df.loc[i]
        
        fcst_df[fcst_df < 0] = 0
        
        return fcst_df
    
    #def exsm_train_score(self, model):
    def exsm_train_score(self, model, seasonal=None):
        if seasonal is not None:
             model = model + "(seasonal)"
        else:
             model = model + "()"    
         
        fcst_split_1 = eval('fcst(self.hist[:-2],self.params, fcst_periods=2, season_periods=self.season_periods).' + model)
        fcst_split_2 = eval('fcst(self.hist[:-4],self.params, fcst_periods=4, season_periods=self.season_periods).' + model)
        fcst_split_3 = eval('fcst(self.hist[:-6],self.params, fcst_periods=6, season_periods=self.season_periods).' + model)
        
        forecasts = [fcst_split_1, fcst_split_2, fcst_split_3]
        
        m1_AE_1 = abs(fcst_split_1[-2:-1] - self.hist[-2:-1])
        m1_AE_2 = abs(fcst_split_2[-4:-3] - self.hist[-4:-3])
        m1_AE_3 = abs(fcst_split_3[-6:-5] - self.hist[-6:-5])
        
        m2_AE_1 = abs(fcst_split_1[-1:] - self.hist[-1:])
        m2_AE_2 = abs(fcst_split_2[-3:-2] - self.hist[-3:-2])
        m2_AE_3 = abs(fcst_split_3[-5:-4] - self.hist[-5:-4])
        
        m3_AE_2 = abs(fcst_split_2[-2:-1] - self.hist[-2:-1])
        m3_AE_3 = abs(fcst_split_3[-4:-3] - self.hist[-4:-3])
        
        m4_AE_2 = abs(fcst_split_2[-1:] - self.hist[-1:])
        m4_AE_3 = abs(fcst_split_3[-3:-2] - self.hist[-3:-2])
        
        #AE_df = m1_AE_1.append(m1_AE_2.append(m1_AE_3.append(m2_AE_1.append(m2_AE_2.append(m2_AE_3.append(m3_AE_2.append(m3_AE_3.append(m4_AE_2.append(m4_AE_3)))))))))
        
        #sdev = AE_df.std()
        
        m1_APE_1 = m1_AE_1 / self.hist[-2:-1]
        m1_APE_2 = m1_AE_2 / self.hist[-4:-3]
        m1_APE_3 = m1_AE_3 / self.hist[-6:-5]
        m2_APE_1 = m2_AE_1 / self.hist[-1:]
        m2_APE_2 = m2_AE_2 / self.hist[-3:-2]
        m2_APE_3 = m2_AE_3 / self.hist[-5:-4]
        m3_APE_2 = m3_AE_2 / self.hist[-2:-1]
        m3_APE_3 = m3_AE_3 / self.hist[-4:-3]
        m4_APE_2 = m4_AE_2 / self.hist[-1:]
        m4_APE_3 = m4_AE_3 / self.hist[-3:-2]
        
        
        m1_MAPE = m1_APE_1.sum()*0.5 + m1_APE_2.sum()*0.3 + m1_APE_3.sum() * 0.2
        m2_MAPE = m2_APE_1.sum()*0.5 + m2_APE_2.sum()*0.3 + m2_APE_3.sum() * 0.2
        m3_MAPE = m3_APE_2.sum()*0.6 + m3_APE_3.sum()*0.4
        m4_MAPE = m4_APE_2.sum()*0.6 + m4_APE_3.sum()*0.4
        
        MAPE = (m1_MAPE*0.4 + m2_MAPE*0.3 + m3_MAPE*0.2 + m4_MAPE*0.1 )
        
        return [MAPE, forecasts] #, sdev
    
    def NN_optimizer(self):
        
        top_MAPE = pd.DataFrame(columns=self.hist.columns,data=999.0,index=['MAPE'])
        top_param = self.params.copy()
        param_adj = 1.0
        model = 'exsm_NN'
        
        cores = mp.cpu_count()
        pool = mp.Pool(cores)
        
        for epoch in range(2):
            print('model:',model, '- epoch',epoch,'start')
            param_opt = top_param.copy()
            param_adj /= 6
            
            par0_opt = param_opt.copy()
            par1_opt = param_opt.copy()
            par2_opt = param_opt.copy()
            par3_opt = param_opt.copy()
            par4_opt = param_opt.copy()
            par5_opt = param_opt.copy()
            par6_opt = param_opt.copy()
            
            par0_opt.loc['Alpha'] -= param_adj * 3
            par1_opt.loc['Alpha'] -= param_adj * 2
            par2_opt.loc['Alpha'] -= param_adj * 1
            par4_opt.loc['Alpha'] += param_adj * 1
            par5_opt.loc['Alpha'] += param_adj * 2
            par6_opt.loc['Alpha'] += param_adj * 3
            
            par0_opt[par0_opt.iloc[:1] > 1] = 1
            par1_opt[par1_opt.iloc[:1] > 1] = 1
            par2_opt[par2_opt.iloc[:1] > 1] = 1
            par4_opt[par4_opt.iloc[:1] > 1] = 1
            par5_opt[par5_opt.iloc[:1] > 1] = 1
            par6_opt[par6_opt.iloc[:1] > 1] = 1
            
            par0_opt[par0_opt.iloc[:1] < 0] = 0
            par1_opt[par1_opt.iloc[:1] < 0] = 0
            par2_opt[par2_opt.iloc[:1] < 0] = 0
            par4_opt[par4_opt.iloc[:1] < 0] = 0
            par5_opt[par5_opt.iloc[:1] < 0] = 0
            par6_opt[par6_opt.iloc[:1] < 0] = 0
            
            p0 = pool.apply_async(fcst(self.hist,par0_opt,season_periods=self.season_periods).exsm_train_score,(model,))
            p1 = pool.apply_async(fcst(self.hist,par1_opt,season_periods=self.season_periods).exsm_train_score,(model,))
            p2 = pool.apply_async(fcst(self.hist,par2_opt,season_periods=self.season_periods).exsm_train_score,(model,))
            p3 = pool.apply_async(fcst(self.hist,par3_opt,season_periods=self.season_periods).exsm_train_score,(model,))
            p4 = pool.apply_async(fcst(self.hist,par4_opt,season_periods=self.season_periods).exsm_train_score,(model,))
            p5 = pool.apply_async(fcst(self.hist,par5_opt,season_periods=self.season_periods).exsm_train_score,(model,))
            p6 = pool.apply_async(fcst(self.hist,par6_opt,season_periods=self.season_periods).exsm_train_score,(model,))
            
            r0 = p0.get()
            r1 = p1.get()
            r2 = p2.get()
            r3 = p3.get()
            r4 = p4.get()
            r5 = p5.get()
            r6 = p6.get()
            
            print('epoch',epoch,'end')
            
            MAPE_df = pd.DataFrame([r0[0],r1[0],r2[0],r3[0],r4[0],r5[0],r6[0]])
            
            min_MAPE = MAPE_df.min()
            top_param_index = np.array(MAPE_df.idxmin())
   
            for i in range(len(top_param_index)):
                 top_par = eval('par' + str(top_param_index[i]) + '_opt')
                 top_param.iloc[0,i] = top_par.iloc[0,i]
                 top_MAPE.iloc[:,i] = min_MAPE[i]
        
        top_fcsts = {}
        for i in range(len(top_param_index)):
            top_r = eval('r' + str(top_param_index[i]) + '[1]')
            top_fcsts[i] = [top_r[0].iloc[:,i], top_r[1].iloc[:,i], top_r[2].iloc[:,i]]
            
        return top_MAPE, top_param, top_fcsts
   
    def NA_optimizer(self,season_add):
        
        top_MAPE = pd.DataFrame(columns=self.hist.columns,data=999.0,index=['MAPE'])
        top_param = self.params.copy()
        param_adj = 1.0
        top_fcsts = {}
        model = 'exsm_NA'
        
        cores = mp.cpu_count()
        pool = mp.Pool(cores)
        
        for epoch in range(2):
            print('model:',model, '- epoch',epoch,'start')
            param_opt = top_param.copy()
            param_adj /= 6
            
            par0_opt = param_opt.copy()
            par1_opt = param_opt.copy()
            par2_opt = param_opt.copy()
            par3_opt = param_opt.copy()
            par4_opt = param_opt.copy()
            par5_opt = param_opt.copy()
            par6_opt = param_opt.copy()
            
            par0_opt.loc['Alpha'] -= param_adj * 3
            par1_opt.loc['Alpha'] -= param_adj * 2
            par2_opt.loc['Alpha'] -= param_adj * 1
            par4_opt.loc['Alpha'] += param_adj * 1
            par5_opt.loc['Alpha'] += param_adj * 2
            par6_opt.loc['Alpha'] += param_adj * 3
            
            for i in range(7):
                 par_opt = eval('par' + str(i) + '_opt')
                 
                 g0_opt = par_opt.copy()
                 g1_opt = par_opt.copy()
                 g2_opt = par_opt.copy()
                 g3_opt = par_opt.copy()
                 g4_opt = par_opt.copy()
                 g5_opt = par_opt.copy()
                 g6_opt = par_opt.copy()
                 
                 g0_opt.loc['Gamma'] -= param_adj * 3
                 g1_opt.loc['Gamma'] -= param_adj * 2
                 g2_opt.loc['Gamma'] -= param_adj * 1
                 g4_opt.loc['Gamma'] += param_adj * 1
                 g5_opt.loc['Gamma'] += param_adj * 2
                 g6_opt.loc['Gamma'] += param_adj * 3
                 
                 g0_opt[g0_opt.iloc[:3] > 1] = 1
                 g1_opt[g1_opt.iloc[:3] > 1] = 1
                 g2_opt[g2_opt.iloc[:3] > 1] = 1
                 g3_opt[g3_opt.iloc[:3] > 1] = 1
                 g4_opt[g4_opt.iloc[:3] > 1] = 1
                 g5_opt[g5_opt.iloc[:3] > 1] = 1
                 g6_opt[g6_opt.iloc[:3] > 1] = 1
                 
                 g0_opt[g0_opt.iloc[:3] < 0] = 0
                 g1_opt[g1_opt.iloc[:3] < 0] = 0
                 g2_opt[g2_opt.iloc[:3] < 0] = 0
                 g3_opt[g3_opt.iloc[:3] < 0] = 0
                 g4_opt[g4_opt.iloc[:3] < 0] = 0
                 g5_opt[g5_opt.iloc[:3] < 0] = 0
                 g6_opt[g6_opt.iloc[:3] < 0] = 0
                 
                 p0 = pool.apply_async(fcst(self.hist,g0_opt,season_periods=self.season_periods).exsm_train_score,(model,season_add))
                 p1 = pool.apply_async(fcst(self.hist,g1_opt,season_periods=self.season_periods).exsm_train_score,(model,season_add))
                 p2 = pool.apply_async(fcst(self.hist,g2_opt,season_periods=self.season_periods).exsm_train_score,(model,season_add))
                 p3 = pool.apply_async(fcst(self.hist,g3_opt,season_periods=self.season_periods).exsm_train_score,(model,season_add))
                 p4 = pool.apply_async(fcst(self.hist,g4_opt,season_periods=self.season_periods).exsm_train_score,(model,season_add))
                 p5 = pool.apply_async(fcst(self.hist,g5_opt,season_periods=self.season_periods).exsm_train_score,(model,season_add))
                 p6 = pool.apply_async(fcst(self.hist,g6_opt,season_periods=self.season_periods).exsm_train_score,(model,season_add))
                 
                 r0 = p0.get()
                 r1 = p1.get()
                 r2 = p2.get()
                 r3 = p3.get()
                 r4 = p4.get()
                 r5 = p5.get()
                 r6 = p6.get()
                 
                 MAPE_df = pd.DataFrame([r0[0],r1[0],r2[0],r3[0],r4[0],r5[0],r6[0]])
                 min_MAPE = MAPE_df.min()
                 top_param_index = np.array(MAPE_df.idxmin())
                 for i in range(len(top_param_index)):
                      if min_MAPE[i] < float(top_MAPE.iloc[:,i]):
                           top_MAPE.iloc[:,i] = min_MAPE[i]
                           top_par = eval('g' + str(top_param_index[i]) + '_opt')
                           top_param.iloc[:3,i] = top_par.iloc[:3,i]
                           top_r = eval('r' + str(top_param_index[i]) + '[1]')
                           top_fcsts[i] = [top_r[0].iloc[:,i], top_r[1].iloc[:,i], top_r[2].iloc[:,i]]

            print('epoch',epoch,'end')
            
        return top_MAPE, top_param, top_fcsts
   
    def NM_optimizer(self,season_mul):
        
        top_MAPE = pd.DataFrame(columns=self.hist.columns,data=999.0,index=['MAPE'])
        top_param = self.params.copy()
        param_adj = 1.0
        top_fcsts = {}
        model = 'exsm_NM'
        
        cores = mp.cpu_count()
        pool = mp.Pool(cores)
        
        for epoch in range(2):
            print('model:',model, '- epoch',epoch,'start')
            param_opt = top_param.copy()
            param_adj /= 6
            
            par0_opt = param_opt.copy()
            par1_opt = param_opt.copy()
            par2_opt = param_opt.copy()
            par3_opt = param_opt.copy()
            par4_opt = param_opt.copy()
            par5_opt = param_opt.copy()
            par6_opt = param_opt.copy()
            
            par0_opt.loc['Alpha'] -= param_adj * 3
            par1_opt.loc['Alpha'] -= param_adj * 2
            par2_opt.loc['Alpha'] -= param_adj * 1
            par4_opt.loc['Alpha'] += param_adj * 1
            par5_opt.loc['Alpha'] += param_adj * 2
            par6_opt.loc['Alpha'] += param_adj * 3
            
            for i in range(7):
                 par_opt = eval('par' + str(i) + '_opt')
                 
                 g0_opt = par_opt.copy()
                 g1_opt = par_opt.copy()
                 g2_opt = par_opt.copy()
                 g3_opt = par_opt.copy()
                 g4_opt = par_opt.copy()
                 g5_opt = par_opt.copy()
                 g6_opt = par_opt.copy()
                 
                 g0_opt.loc['Gamma'] -= param_adj * 3
                 g1_opt.loc['Gamma'] -= param_adj * 2
                 g2_opt.loc['Gamma'] -= param_adj * 1
                 g4_opt.loc['Gamma'] += param_adj * 1
                 g5_opt.loc['Gamma'] += param_adj * 2
                 g6_opt.loc['Gamma'] += param_adj * 3
                 
                 g0_opt[g0_opt.iloc[:3] > 1] = 1
                 g1_opt[g1_opt.iloc[:3] > 1] = 1
                 g2_opt[g2_opt.iloc[:3] > 1] = 1
                 g3_opt[g3_opt.iloc[:3] > 1] = 1
                 g4_opt[g4_opt.iloc[:3] > 1] = 1
                 g5_opt[g5_opt.iloc[:3] > 1] = 1
                 g6_opt[g6_opt.iloc[:3] > 1] = 1
                 
                 g0_opt[g0_opt.iloc[:3] < 0] = 0
                 g1_opt[g1_opt.iloc[:3] < 0] = 0
                 g2_opt[g2_opt.iloc[:3] < 0] = 0
                 g3_opt[g3_opt.iloc[:3] < 0] = 0
                 g4_opt[g4_opt.iloc[:3] < 0] = 0
                 g5_opt[g5_opt.iloc[:3] < 0] = 0
                 g6_opt[g6_opt.iloc[:3] < 0] = 0
                 
                 p0 = pool.apply_async(fcst(self.hist,g0_opt,season_periods=self.season_periods).exsm_train_score,(model,season_mul))
                 p1 = pool.apply_async(fcst(self.hist,g1_opt,season_periods=self.season_periods).exsm_train_score,(model,season_mul))
                 p2 = pool.apply_async(fcst(self.hist,g2_opt,season_periods=self.season_periods).exsm_train_score,(model,season_mul))
                 p3 = pool.apply_async(fcst(self.hist,g3_opt,season_periods=self.season_periods).exsm_train_score,(model,season_mul))
                 p4 = pool.apply_async(fcst(self.hist,g4_opt,season_periods=self.season_periods).exsm_train_score,(model,season_mul))
                 p5 = pool.apply_async(fcst(self.hist,g5_opt,season_periods=self.season_periods).exsm_train_score,(model,season_mul))
                 p6 = pool.apply_async(fcst(self.hist,g6_opt,season_periods=self.season_periods).exsm_train_score,(model,season_mul))
                 
                 r0 = p0.get()
                 r1 = p1.get()
                 r2 = p2.get()
                 r3 = p3.get()
                 r4 = p4.get()
                 r5 = p5.get()
                 r6 = p6.get()
                 
                 MAPE_df = pd.DataFrame([r0[0],r1[0],r2[0],r3[0],r4[0],r5[0],r6[0]])
                 min_MAPE = MAPE_df.min()
                 top_param_index = np.array(MAPE_df.idxmin())
                 
                 for i in range(len(top_param_index)):
                      if min_MAPE[i] < float(top_MAPE.iloc[:,i]):
                           top_MAPE.iloc[:,i] = min_MAPE[i]
                           top_par = eval('g' + str(top_param_index[i]) + '_opt')
                           top_param.iloc[:3,i] = top_par.iloc[:3,i]
                           top_r = eval('r' + str(top_param_index[i]) + '[1]')
                           top_fcsts[i] = [top_r[0].iloc[:,i], top_r[1].iloc[:,i], top_r[2].iloc[:,i]]

            print('epoch',epoch,'end')
            
        return top_MAPE, top_param, top_fcsts
   
    def LN_optimizer(self):
        
        top_MAPE = pd.DataFrame(columns=self.hist.columns,data=999.0,index=['MAPE'])
        top_param = self.params.copy()
        param_adj = 1
        top_fcsts = {}
        model = 'exsm_LN'
        
        cores = mp.cpu_count()
        pool = mp.Pool(cores)
        
        for epoch in range(2):
            print('model:',model, '- epoch',epoch,'start')
            param_opt = top_param.copy()
            param_adj /= 6
            
            par0_opt = param_opt.copy()
            par1_opt = param_opt.copy()
            par2_opt = param_opt.copy()
            par3_opt = param_opt.copy()
            par4_opt = param_opt.copy()
            par5_opt = param_opt.copy()
            par6_opt = param_opt.copy()
            
            par0_opt.loc['Alpha'] -= param_adj * 3
            par1_opt.loc['Alpha'] -= param_adj * 2
            par2_opt.loc['Alpha'] -= param_adj * 1
            par4_opt.loc['Alpha'] += param_adj * 1
            par5_opt.loc['Alpha'] += param_adj * 2
            par6_opt.loc['Alpha'] += param_adj * 3
            
            for i in range(7):
                 par_opt = eval('par' + str(i) + '_opt')
                 
                 b0_opt = par_opt.copy()
                 b1_opt = par_opt.copy()
                 b2_opt = par_opt.copy()
                 b3_opt = par_opt.copy()
                 b4_opt = par_opt.copy()
                 b5_opt = par_opt.copy()
                 b6_opt = par_opt.copy()
                 
                 b0_opt.loc['Beta'] -= param_adj * 3
                 b1_opt.loc['Beta'] -= param_adj * 2
                 b2_opt.loc['Beta'] -= param_adj * 1
                 b4_opt.loc['Beta'] += param_adj * 1
                 b5_opt.loc['Beta'] += param_adj * 2
                 b6_opt.loc['Beta'] += param_adj * 3
                 
                 b0_opt[b0_opt.iloc[:3] > 1] = 1
                 b1_opt[b1_opt.iloc[:3] > 1] = 1
                 b2_opt[b2_opt.iloc[:3] > 1] = 1
                 b3_opt[b3_opt.iloc[:3] > 1] = 1
                 b4_opt[b4_opt.iloc[:3] > 1] = 1
                 b5_opt[b5_opt.iloc[:3] > 1] = 1
                 b6_opt[b6_opt.iloc[:3] > 1] = 1
                 
                 b0_opt[b0_opt.iloc[:3] < 0] = 0
                 b1_opt[b1_opt.iloc[:3] < 0] = 0
                 b2_opt[b2_opt.iloc[:3] < 0] = 0
                 b3_opt[b3_opt.iloc[:3] < 0] = 0
                 b4_opt[b4_opt.iloc[:3] < 0] = 0
                 b5_opt[b5_opt.iloc[:3] < 0] = 0
                 b6_opt[b6_opt.iloc[:3] < 0] = 0
                 
                 p0 = pool.apply_async(fcst(self.hist,b0_opt,season_periods=self.season_periods).exsm_train_score,(model,))
                 p1 = pool.apply_async(fcst(self.hist,b1_opt,season_periods=self.season_periods).exsm_train_score,(model,))
                 p2 = pool.apply_async(fcst(self.hist,b2_opt,season_periods=self.season_periods).exsm_train_score,(model,))
                 p3 = pool.apply_async(fcst(self.hist,b3_opt,season_periods=self.season_periods).exsm_train_score,(model,))
                 p4 = pool.apply_async(fcst(self.hist,b4_opt,season_periods=self.season_periods).exsm_train_score,(model,))
                 p5 = pool.apply_async(fcst(self.hist,b5_opt,season_periods=self.season_periods).exsm_train_score,(model,))
                 p6 = pool.apply_async(fcst(self.hist,b6_opt,season_periods=self.season_periods).exsm_train_score,(model,))
                 
                 r0 = p0.get()
                 r1 = p1.get()
                 r2 = p2.get()
                 r3 = p3.get()
                 r4 = p4.get()
                 r5 = p5.get()
                 r6 = p6.get()
                 
                 MAPE_df = pd.DataFrame([r0[0],r1[0],r2[0],r3[0],r4[0],r5[0],r6[0]])
                 min_MAPE = MAPE_df.min()
                 top_param_index = np.array(MAPE_df.idxmin())
                 for i in range(len(top_param_index)):
                      if min_MAPE[i] < float(top_MAPE.iloc[:,i]):
                           top_MAPE.iloc[:,i] = min_MAPE[i]
                           top_par = eval('b' + str(top_param_index[i]) + '_opt')
                           top_param.iloc[:3,i] = top_par.iloc[:3,i]
                           top_r = eval('r' + str(top_param_index[i]) + '[1]')
                           top_fcsts[i] = [top_r[0].iloc[:,i], top_r[1].iloc[:,i], top_r[2].iloc[:,i]]

            print('epoch',epoch,'end')
            
        return top_MAPE, top_param, top_fcsts
   
    def LA_optimizer(self,season_add):
        
        top_MAPE = pd.DataFrame(columns=self.hist.columns,data=999.0,index=['MAPE'])
        top_param = self.params.copy()
        param_adj = 1.0
        top_fcsts = {}
        model = 'exsm_LA'
        
        cores = mp.cpu_count()
        pool = mp.Pool(cores)
        
        for epoch in range(2):
            print('model:',model, '- epoch',epoch,'start')
            param_opt = top_param.copy()
            param_adj /= 6
            
            par0_opt = param_opt.copy()
            par1_opt = param_opt.copy()
            par2_opt = param_opt.copy()
            par3_opt = param_opt.copy()
            par4_opt = param_opt.copy()
            par5_opt = param_opt.copy()
            par6_opt = param_opt.copy()
            
            par0_opt.loc['Alpha'] -= param_adj * 3
            par1_opt.loc['Alpha'] -= param_adj * 2
            par2_opt.loc['Alpha'] -= param_adj * 1
            par4_opt.loc['Alpha'] += param_adj * 1
            par5_opt.loc['Alpha'] += param_adj * 2
            par6_opt.loc['Alpha'] += param_adj * 3
            
            for i in range(7):
                 par_opt = eval('par' + str(i) + '_opt')
                 
                 b0_opt = par_opt.copy()
                 b1_opt = par_opt.copy()
                 b2_opt = par_opt.copy()
                 b3_opt = par_opt.copy()
                 b4_opt = par_opt.copy()
                 b5_opt = par_opt.copy()
                 b6_opt = par_opt.copy()
                 
                 b0_opt.loc['Beta'] -= param_adj * 3
                 b1_opt.loc['Beta'] -= param_adj * 2
                 b2_opt.loc['Beta'] -= param_adj * 1
                 b4_opt.loc['Beta'] += param_adj * 1
                 b5_opt.loc['Beta'] += param_adj * 2
                 b6_opt.loc['Beta'] += param_adj * 3
                 
                 for i in range(7):
                      b_opt = eval('b' + str(i) + '_opt')
                      
                      g0_opt = b_opt.copy()
                      g1_opt = b_opt.copy()
                      g2_opt = b_opt.copy()
                      g3_opt = b_opt.copy()
                      g4_opt = b_opt.copy()
                      g5_opt = b_opt.copy()
                      g6_opt = b_opt.copy()
                      
                      g0_opt.loc['Gamma'] -= param_adj * 3
                      g1_opt.loc['Gamma'] -= param_adj * 2
                      g2_opt.loc['Gamma'] -= param_adj * 1
                      g4_opt.loc['Gamma'] += param_adj * 1
                      g5_opt.loc['Gamma'] += param_adj * 2
                      g6_opt.loc['Gamma'] += param_adj * 3
                      
                      g0_opt[g0_opt.iloc[:3] > 1] = 1
                      g1_opt[g1_opt.iloc[:3] > 1] = 1
                      g2_opt[g2_opt.iloc[:3] > 1] = 1
                      g3_opt[g3_opt.iloc[:3] > 1] = 1
                      g4_opt[g4_opt.iloc[:3] > 1] = 1
                      g5_opt[g5_opt.iloc[:3] > 1] = 1
                      g6_opt[g6_opt.iloc[:3] > 1] = 1
                      
                      g0_opt[g0_opt.iloc[:3] < 0] = 0
                      g1_opt[g1_opt.iloc[:3] < 0] = 0
                      g2_opt[g2_opt.iloc[:3] < 0] = 0
                      g3_opt[g3_opt.iloc[:3] < 0] = 0
                      g4_opt[g4_opt.iloc[:3] < 0] = 0
                      g5_opt[g5_opt.iloc[:3] < 0] = 0
                      g6_opt[g6_opt.iloc[:3] < 0] = 0
                 
                      p0 = pool.apply_async(fcst(self.hist,g0_opt,season_periods=self.season_periods).exsm_train_score,(model,season_add))
                      p1 = pool.apply_async(fcst(self.hist,g1_opt,season_periods=self.season_periods).exsm_train_score,(model,season_add))
                      p2 = pool.apply_async(fcst(self.hist,g2_opt,season_periods=self.season_periods).exsm_train_score,(model,season_add))
                      p3 = pool.apply_async(fcst(self.hist,g3_opt,season_periods=self.season_periods).exsm_train_score,(model,season_add))
                      p4 = pool.apply_async(fcst(self.hist,g4_opt,season_periods=self.season_periods).exsm_train_score,(model,season_add))
                      p5 = pool.apply_async(fcst(self.hist,g5_opt,season_periods=self.season_periods).exsm_train_score,(model,season_add))
                      p6 = pool.apply_async(fcst(self.hist,g6_opt,season_periods=self.season_periods).exsm_train_score,(model,season_add))
                 
                      r0 = p0.get()
                      r1 = p1.get()
                      r2 = p2.get()
                      r3 = p3.get()
                      r4 = p4.get()
                      r5 = p5.get()
                      r6 = p6.get()
                 
                      MAPE_df = pd.DataFrame([r0[0],r1[0],r2[0],r3[0],r4[0],r5[0],r6[0]])
                      min_MAPE = MAPE_df.min()
                      top_param_index = np.array(MAPE_df.idxmin())
                      for i in range(len(top_param_index)):
                           if min_MAPE[i] < float(top_MAPE.iloc[:,i]):
                                top_MAPE.iloc[:,i] = min_MAPE[i]
                                top_par = eval('g' + str(top_param_index[i]) + '_opt')
                                top_param.iloc[:3,i] = top_par.iloc[:3,i]
                                top_r = eval('r' + str(top_param_index[i]) + '[1]')
                                top_fcsts[i] = [top_r[0].iloc[:,i], top_r[1].iloc[:,i], top_r[2].iloc[:,i]]

            print('epoch',epoch,'end')
            
        return top_MAPE, top_param, top_fcsts
   
    def LM_optimizer(self,season_mul):
        
        top_MAPE = pd.DataFrame(columns=self.hist.columns,data=999.0,index=['MAPE'])
        top_param = self.params.copy()
        param_adj = 1.0
        top_fcsts = {}
        model = 'exsm_LM'
        
        cores = mp.cpu_count()
        pool = mp.Pool(cores)
        
        for epoch in range(2):
            print('model:',model, '- epoch',epoch,'start')
            param_opt = top_param.copy()
            param_adj /= 6
            
            par0_opt = param_opt.copy()
            par1_opt = param_opt.copy()
            par2_opt = param_opt.copy()
            par3_opt = param_opt.copy()
            par4_opt = param_opt.copy()
            par5_opt = param_opt.copy()
            par6_opt = param_opt.copy()
            
            par0_opt.loc['Alpha'] -= param_adj * 3
            par1_opt.loc['Alpha'] -= param_adj * 2
            par2_opt.loc['Alpha'] -= param_adj * 1
            par4_opt.loc['Alpha'] += param_adj * 1
            par5_opt.loc['Alpha'] += param_adj * 2
            par6_opt.loc['Alpha'] += param_adj * 3
            
            for i in range(7):
                 par_opt = eval('par' + str(i) + '_opt')
                 
                 b0_opt = par_opt.copy()
                 b1_opt = par_opt.copy()
                 b2_opt = par_opt.copy()
                 b3_opt = par_opt.copy()
                 b4_opt = par_opt.copy()
                 b5_opt = par_opt.copy()
                 b6_opt = par_opt.copy()
                 
                 b0_opt.loc['Beta'] -= param_adj * 3
                 b1_opt.loc['Beta'] -= param_adj * 2
                 b2_opt.loc['Beta'] -= param_adj * 1
                 b4_opt.loc['Beta'] += param_adj * 1
                 b5_opt.loc['Beta'] += param_adj * 2
                 b6_opt.loc['Beta'] += param_adj * 3
                 
                 for i in range(7):
                      b_opt = eval('b' + str(i) + '_opt')
                      
                      g0_opt = b_opt.copy()
                      g1_opt = b_opt.copy()
                      g2_opt = b_opt.copy()
                      g3_opt = b_opt.copy()
                      g4_opt = b_opt.copy()
                      g5_opt = b_opt.copy()
                      g6_opt = b_opt.copy()
                      
                      g0_opt.loc['Gamma'] -= param_adj * 3
                      g1_opt.loc['Gamma'] -= param_adj * 2
                      g2_opt.loc['Gamma'] -= param_adj * 1
                      g4_opt.loc['Gamma'] += param_adj * 1
                      g5_opt.loc['Gamma'] += param_adj * 2
                      g6_opt.loc['Gamma'] += param_adj * 3
                      
                      g0_opt[g0_opt.iloc[:3] > 1] = 1
                      g1_opt[g1_opt.iloc[:3] > 1] = 1
                      g2_opt[g2_opt.iloc[:3] > 1] = 1
                      g3_opt[g3_opt.iloc[:3] > 1] = 1
                      g4_opt[g4_opt.iloc[:3] > 1] = 1
                      g5_opt[g5_opt.iloc[:3] > 1] = 1
                      g6_opt[g6_opt.iloc[:3] > 1] = 1
                      
                      g0_opt[g0_opt.iloc[:3] < 0] = 0
                      g1_opt[g1_opt.iloc[:3] < 0] = 0
                      g2_opt[g2_opt.iloc[:3] < 0] = 0
                      g3_opt[g3_opt.iloc[:3] < 0] = 0
                      g4_opt[g4_opt.iloc[:3] < 0] = 0
                      g5_opt[g5_opt.iloc[:3] < 0] = 0
                      g6_opt[g6_opt.iloc[:3] < 0] = 0
                 
                      p0 = pool.apply_async(fcst(self.hist,g0_opt,season_periods=self.season_periods).exsm_train_score,(model,season_mul))
                      p1 = pool.apply_async(fcst(self.hist,g1_opt,season_periods=self.season_periods).exsm_train_score,(model,season_mul))
                      p2 = pool.apply_async(fcst(self.hist,g2_opt,season_periods=self.season_periods).exsm_train_score,(model,season_mul))
                      p3 = pool.apply_async(fcst(self.hist,g3_opt,season_periods=self.season_periods).exsm_train_score,(model,season_mul))
                      p4 = pool.apply_async(fcst(self.hist,g4_opt,season_periods=self.season_periods).exsm_train_score,(model,season_mul))
                      p5 = pool.apply_async(fcst(self.hist,g5_opt,season_periods=self.season_periods).exsm_train_score,(model,season_mul))
                      p6 = pool.apply_async(fcst(self.hist,g6_opt,season_periods=self.season_periods).exsm_train_score,(model,season_mul))
                 
                      r0 = p0.get()
                      r1 = p1.get()
                      r2 = p2.get()
                      r3 = p3.get()
                      r4 = p4.get()
                      r5 = p5.get()
                      r6 = p6.get()
                 
                      MAPE_df = pd.DataFrame([r0[0],r1[0],r2[0],r3[0],r4[0],r5[0],r6[0]])
                      min_MAPE = MAPE_df.min()
                      top_param_index = np.array(MAPE_df.idxmin())
                      for i in range(len(top_param_index)):
                           if min_MAPE[i] < float(top_MAPE.iloc[:,i]):
                                top_MAPE.iloc[:,i] = min_MAPE[i]
                                top_par = eval('g' + str(top_param_index[i]) + '_opt')
                                top_param.iloc[:3,i] = top_par.iloc[:3,i]
                                top_r = eval('r' + str(top_param_index[i]) + '[1]')
                                top_fcsts[i] = [top_r[0].iloc[:,i], top_r[1].iloc[:,i], top_r[2].iloc[:,i]]

            print('epoch',epoch,'end')
            
        return top_MAPE, top_param, top_fcsts
   
    def Phi_optimizer(self,model,top_MAPE,top_param,season):
        
        #top_MAPE = pd.DataFrame(columns=self.hist.columns,data=999.0,index=['MAPE'])
        #top_param = self.params.copy()
        param_adj = 1.0
        #model = 'exsm_NN'
        
        cores = mp.cpu_count()
        pool = mp.Pool(cores)
        
        for epoch in range(2):
            print('Phi - model:',model, '- epoch',epoch,'start')
            param_opt = top_param.copy()
            param_adj /= 6
            
            par0_opt = param_opt.copy()
            par1_opt = param_opt.copy()
            par2_opt = param_opt.copy()
            par3_opt = param_opt.copy()
            par4_opt = param_opt.copy()
            par5_opt = param_opt.copy()
            par6_opt = param_opt.copy()
            
            par0_opt.loc['Phi'] -= param_adj * 3
            par1_opt.loc['Phi'] -= param_adj * 2
            par2_opt.loc['Phi'] -= param_adj * 1
            par4_opt.loc['Phi'] += param_adj * 1
            par5_opt.loc['Phi'] += param_adj * 2
            par6_opt.loc['Phi'] += param_adj * 3
            
            par0_opt[par0_opt.iloc[:4] > 1] = 1
            par1_opt[par1_opt.iloc[:4] > 1] = 1
            par2_opt[par2_opt.iloc[:4] > 1] = 1
            par4_opt[par4_opt.iloc[:4] > 1] = 1
            par5_opt[par5_opt.iloc[:4] > 1] = 1
            par6_opt[par6_opt.iloc[:4] > 1] = 1
            
            par0_opt[par0_opt.iloc[:4] < 0] = 0
            par1_opt[par1_opt.iloc[:4] < 0] = 0
            par2_opt[par2_opt.iloc[:4] < 0] = 0
            par4_opt[par4_opt.iloc[:4] < 0] = 0
            par5_opt[par5_opt.iloc[:4] < 0] = 0
            par6_opt[par6_opt.iloc[:4] < 0] = 0
            
            if model == 'exsm_LN':
                 
                 p0 = pool.apply_async(fcst(self.hist,par0_opt,season_periods=self.season_periods).exsm_train_score,(model,))
                 p1 = pool.apply_async(fcst(self.hist,par1_opt,season_periods=self.season_periods).exsm_train_score,(model,))
                 p2 = pool.apply_async(fcst(self.hist,par2_opt,season_periods=self.season_periods).exsm_train_score,(model,))
                 p3 = pool.apply_async(fcst(self.hist,par3_opt,season_periods=self.season_periods).exsm_train_score,(model,))
                 p4 = pool.apply_async(fcst(self.hist,par4_opt,season_periods=self.season_periods).exsm_train_score,(model,))
                 p5 = pool.apply_async(fcst(self.hist,par5_opt,season_periods=self.season_periods).exsm_train_score,(model,))
                 p6 = pool.apply_async(fcst(self.hist,par6_opt,season_periods=self.season_periods).exsm_train_score,(model,))
            else:
                 p0 = pool.apply_async(fcst(self.hist,par0_opt,season_periods=self.season_periods).exsm_train_score,(model,season))
                 p1 = pool.apply_async(fcst(self.hist,par1_opt,season_periods=self.season_periods).exsm_train_score,(model,season))
                 p2 = pool.apply_async(fcst(self.hist,par2_opt,season_periods=self.season_periods).exsm_train_score,(model,season))
                 p3 = pool.apply_async(fcst(self.hist,par3_opt,season_periods=self.season_periods).exsm_train_score,(model,season))
                 p4 = pool.apply_async(fcst(self.hist,par4_opt,season_periods=self.season_periods).exsm_train_score,(model,season))
                 p5 = pool.apply_async(fcst(self.hist,par5_opt,season_periods=self.season_periods).exsm_train_score,(model,season))
                 p6 = pool.apply_async(fcst(self.hist,par6_opt,season_periods=self.season_periods).exsm_train_score,(model,season))
            
            r0 = p0.get()
            r1 = p1.get()
            r2 = p2.get()
            r3 = p3.get()
            r4 = p4.get()
            r5 = p5.get()
            r6 = p6.get()
            
            print('epoch',epoch,'end')
            
            MAPE_df = pd.DataFrame([r0[0],r1[0],r2[0],r3[0],r4[0],r5[0],r6[0]])
            
            min_MAPE = MAPE_df.min()
            top_param_index = np.array(MAPE_df.idxmin())
   
            for i in range(len(top_param_index)):
                 top_par = eval('par' + str(top_param_index[i]) + '_opt')
                 top_param.iloc[0,i] = top_par.iloc[0,i]
                 top_MAPE.iloc[:,i] = min_MAPE[i]
        
        top_fcsts = {}
        for i in range(len(top_param_index)):
            top_r = eval('r' + str(top_param_index[i]) + '[1]')
            top_fcsts[i] = [top_r[0].iloc[:,i], top_r[1].iloc[:,i], top_r[2].iloc[:,i]]
            
        return top_MAPE, top_param, top_fcsts
     
    def exsm_optimizer(self,season_add,season_mul):
         
         NN_MAPE, NN_param, NN_fcsts = self.NN_optimizer()
         NA_MAPE, NA_param, NA_fcsts = self.NA_optimizer(season_add)
         NM_MAPE, NM_param, NM_fcsts = self.NM_optimizer(season_mul)
         LN_MAPE, LN_param, LN_fcsts = self.LN_optimizer()
         LN_MAPE, LN_param, LN_fcsts = self.Phi_optimizer('exsm_LN',LN_MAPE,LN_param,None)
         LA_MAPE, LA_param, LA_fcsts = self.LA_optimizer(season_add)
         LA_MAPE, LA_param, LA_fcsts = self.Phi_optimizer('exsm_LA',LA_MAPE,LA_param,season_add)
         LM_MAPE, LM_param, LM_fcsts = self.LM_optimizer(season_mul)
         LM_MAPE, LM_param, LM_fcsts = self.Phi_optimizer('exsm_LM',LM_MAPE,LM_param,season_mul)
         
         top_MAPE = {'NN':NN_MAPE, 'NA':NA_MAPE, 'NM':NM_MAPE, 'LN':LN_MAPE, 'LA':LA_MAPE, 'LM':LM_MAPE}
         top_param = {'NN':NN_param, 'NA':NA_param, 'NM':NM_param, 'LN':LN_param, 'LA':LA_param, 'LM':LM_param}
         top_fcsts = {'NN':NN_fcsts, 'NA':NA_fcsts, 'NM':NM_fcsts, 'LN':LN_fcsts, 'LA':LA_fcsts, 'LM':LM_fcsts}
         
         return [top_MAPE, top_param, top_fcsts]
    

    def exsm_test_split(self, model, params, seasonal=None):
        if seasonal is not None:
             model = model + "(seasonal)"
        else:
             model = model + "()"    
         
        fcst_split_1 = eval('fcst(self.hist[:-1],params, fcst_periods=1, season_periods=self.season_periods).' + model)
        fcst_split_2 = eval('fcst(self.hist[:-3],params, fcst_periods=3, season_periods=self.season_periods).' + model)
        fcst_split_3 = eval('fcst(self.hist[:-5],params, fcst_periods=5, season_periods=self.season_periods).' + model)
        
        forecasts = [fcst_split_1, fcst_split_2, fcst_split_3]
        return forecasts
   
    def model_test_score(self, forecasts):
         
        fcst_split_1, fcst_split_2, fcst_split_3 = forecasts
        
        m1_AE_1 = abs(fcst_split_1[-1:] - self.hist[-1:])
        m1_AE_2 = abs(fcst_split_2[-3:-2] - self.hist[-3:-2])
        m1_AE_3 = abs(fcst_split_3[-5:-4] - self.hist[-5:-4])
        
        m2_AE_2 = abs(fcst_split_2[-2:-1] - self.hist[-2:-1])
        m2_AE_3 = abs(fcst_split_3[-4:-3] - self.hist[-4:-3])
        
        m3_AE_2 = abs(fcst_split_2[-1:] - self.hist[-1:])
        m3_AE_3 = abs(fcst_split_3[-3:-2] - self.hist[-3:-2])
        
        m4_AE_3 = abs(fcst_split_3[-2:-1] - self.hist[-2:-1])
        
        AE_df = m1_AE_1.append(m1_AE_2.append(m1_AE_3.append(m2_AE_2.append(m2_AE_3.append(m3_AE_2.append(m3_AE_3.append(m4_AE_3)))))))
        
        sdev = AE_df.std()
        
        m1_APE_1 = m1_AE_1 / self.hist[-1:]
        m1_APE_2 = m1_AE_2 / self.hist[-3:-2]
        m1_APE_3 = m1_AE_3 / self.hist[-5:-4]
        m2_APE_2 = m2_AE_2 / self.hist[-2:-1]
        m2_APE_3 = m2_AE_3 / self.hist[-4:-3]
        m3_APE_2 = m3_AE_2 / self.hist[-1:]
        m3_APE_3 = m3_AE_3 / self.hist[-3:-2]
        m4_APE_3 = m4_AE_3 / self.hist[-2:-1]
        
        
        m1_MAPE = m1_APE_1.sum()*0.5 + m1_APE_2.sum()*0.3 + m1_APE_3.sum() * 0.2
        m2_MAPE = m2_APE_2.sum()*0.6 + m2_APE_3.sum()*0.4
        m3_MAPE = m3_APE_2.sum()*0.6 + m3_APE_3.sum()*0.4
        m4_MAPE = m4_APE_3.sum()
        
        MAPE = (m1_MAPE*0.4 + m2_MAPE*0.3 + m3_MAPE*0.2 + m4_MAPE*0.1 )
        
        return [MAPE, sdev]
   
    def exsm_test_score(self, params, season_add, season_mul):         
         
         cores = mp.cpu_count()
         pool = mp.Pool(cores)
         
         p0 = pool.apply_async(self.exsm_test_split,('exsm_NN',params['NN']))
         p1 = pool.apply_async(self.exsm_test_split,('exsm_NA',params['NA'],season_add))
         p2 = pool.apply_async(self.exsm_test_split,('exsm_NM',params['NM'],season_mul))
         p3 = pool.apply_async(self.exsm_test_split,('exsm_LN',params['LN']))
         p4 = pool.apply_async(self.exsm_test_split,('exsm_LA',params['LA'],season_add))
         p5 = pool.apply_async(self.exsm_test_split,('exsm_LM',params['LM'],season_mul))
         
         NN_split = p0.get()
         NA_split = p1.get()
         NM_split = p2.get()
         LN_split = p3.get()
         LA_split = p4.get()
         LM_split = p5.get()
         
         NN_score = self.model_test_score(NN_split)
         NA_score = self.model_test_score(NA_split)
         NM_score = self.model_test_score(NM_split)
         LN_score = self.model_test_score(LN_split)
         LA_score = self.model_test_score(LA_split)
         LM_score = self.model_test_score(LM_split)
         
         scores = {'NN':NN_score, 'NA':NA_score, 'NM':NM_score, 'LN':LN_score, 'LA':LA_score, 'LM':LM_score}
         fcsts = {'NN':NN_split, 'NA':NA_split, 'NM':NM_split, 'LN':LN_split, 'LA':LA_split, 'LM':LM_split}
         
         #NN_MAPE = NN_score[0].to_frame('NN').transpose()
         #NM_MAPE = NM_score[0].to_frame('NM').transpose()
         #LN_MAPE = LN_score[0].to_frame('LN').transpose()
         #LA_MAPE = LA_score[0].to_frame('LA').transpose()
         #LM_MAPE = LM_score[0].to_frame('LM').transpose()
         
         #MAPE = NN_MAPE.append(NA_MAPE.append(NM_MAPE.append(LN_MAPE.append(LA_MAPE.append(LM_MAPE)))))
         
         return scores, fcsts
        
    def top_model(self, season_add, season_mul):
         
         models = self.exsm_optimizer(season_add,season_mul)
         
         top_params = models[1]
         
         exsm_score, exsm_fcsts = self.exsm_test_score(models[1], season_add, season_mul)
         
         NN_MAPE = exsm_score['NN'][0].to_frame('NN').transpose()
         NA_MAPE = exsm_score['NA'][0].to_frame('NA').transpose()
         NM_MAPE = exsm_score['NM'][0].to_frame('NM').transpose()
         LN_MAPE = exsm_score['LN'][0].to_frame('LN').transpose()
         LA_MAPE = exsm_score['LA'][0].to_frame('LA').transpose()
         LM_MAPE = exsm_score['LM'][0].to_frame('LM').transpose()
         
         MAPE = NN_MAPE.append(NA_MAPE.append(NM_MAPE.append(LN_MAPE.append(LA_MAPE.append(LM_MAPE)))))
         
         NN_sdev = exsm_score['NN'][1].to_frame('NN').transpose()
         NA_sdev = exsm_score['NA'][1].to_frame('NA').transpose()
         NM_sdev = exsm_score['NM'][1].to_frame('NM').transpose()
         LN_sdev = exsm_score['LN'][1].to_frame('LN').transpose()
         LA_sdev = exsm_score['LA'][1].to_frame('LA').transpose()
         LM_sdev = exsm_score['LM'][1].to_frame('LM').transpose()
         
         sdev = NN_sdev.append(NA_sdev.append(NM_sdev.append(LN_sdev.append(LA_sdev.append(LM_sdev)))))
         
         blend_model = []
         blend_split = [self.hist.copy(),self.hist.copy(),self.hist.copy()]
         for column in range(len(MAPE.columns)):
              M1 = MAPE.iloc[:,column].nsmallest(2).index[0]
              M2 = MAPE.iloc[:,column].nsmallest(2).index[1]
              blend_model.append(M1+'_'+M2)
              #print(blend_model[column])
              
              blend_split[0].iloc[:,column] = exsm_fcsts[M1][0].iloc[:,column] * 0.7 + exsm_fcsts[M2][0].iloc[:,column] * 0.3
              blend_split[1].iloc[:,column] = exsm_fcsts[M1][1].iloc[:,column] * 0.7 + exsm_fcsts[M2][1].iloc[:,column] * 0.3
              blend_split[2].iloc[:,column] = exsm_fcsts[M1][2].iloc[:,column] * 0.7 + exsm_fcsts[M2][2].iloc[:,column] * 0.3
              
         blend_score =  self.model_test_score(blend_split)
         
         MAPE = MAPE.append(blend_score[0].to_frame('blend').transpose())
         sdev = sdev.append(blend_score[1].to_frame('blend').transpose())
         
         best_model = MAPE.idxmin()
         
         best_sdev = []
         for i in range(len(sdev.columns)):
              best_sdev.append(sdev.loc[best_model[i]][i])
              #if best_model[i] == 'blend':
              #     sdev.iloc[:,i] = blend_score[1].iloc[:,i]
         
         for idx, model in enumerate(best_model):
              if model == 'blend':
                   best_model[idx] = blend_model[idx]                 
         
         best_model = best_model.to_frame('model')
         best_model['MAPE'] = MAPE.min()
         best_model['sdev'] = best_sdev
         return best_model, top_params
    
    def wrapper(self, func, *args):
         try:
              return func(*args)
         except:
              print('forecast not selected',func)
    
    def final_fcst(self, model, params, season_add, season_mul):
         
         NN_list = list(model[model['model'].str.contains('NN')].index)
         NA_list = list(model[model['model'].str.contains('NA')].index)
         NM_list = list(model[model['model'].str.contains('NM')].index)
         LN_list = list(model[model['model'].str.contains('LN')].index)
         LA_list = list(model[model['model'].str.contains('LA')].index)
         LM_list = list(model[model['model'].str.contains('LM')].index)
         
         NN_hist = self.hist[NN_list]
         NA_hist = self.hist[NA_list]
         NM_hist = self.hist[NM_list]
         LN_hist = self.hist[LN_list]
         LA_hist = self.hist[LA_list]
         LM_hist = self.hist[LM_list]
         
         NN_param = params['NN'][NN_list]
         NA_param = params['NA'][NA_list]
         NM_param = params['NM'][NM_list]
         LN_param = params['LN'][LN_list]
         LA_param = params['LA'][LA_list]
         LM_param = params['LM'][LM_list]
         
         SA_NA = season_add[NA_list]
         SA_LA = season_add[LA_list]
         SM_NM = season_mul[NM_list]
         SM_LM = season_mul[LM_list]
         
         cores = mp.cpu_count()
         pool = mp.Pool(cores)
         
         p0 = pool.apply_async(self.wrapper,args=(fcst(NN_hist,NN_param,self.fcst_periods,self.season_periods).exsm_NN,))
         p1 = pool.apply_async(self.wrapper,args=(fcst(NA_hist,NA_param,self.fcst_periods,self.season_periods).exsm_NA,SA_NA))
         p2 = pool.apply_async(self.wrapper,args=(fcst(NM_hist,NM_param,self.fcst_periods,self.season_periods).exsm_NM,SM_NM))
         p3 = pool.apply_async(self.wrapper,args=(fcst(LN_hist,LN_param,self.fcst_periods,self.season_periods).exsm_LN,))
         p4 = pool.apply_async(self.wrapper,args=(fcst(LA_hist,LA_param,self.fcst_periods,self.season_periods).exsm_LA,SA_LA))
         p5 = pool.apply_async(self.wrapper,args=(fcst(LM_hist,LM_param,self.fcst_periods,self.season_periods).exsm_LM,SM_LM))
         
         #p0 = pool.apply_async(fcst(NN_hist,NN_param,self.fcst_periods,self.season_periods).exsm_NN,())
         #p1 = pool.apply_async(fcst(NA_hist,NA_param,self.fcst_periods,self.season_periods).exsm_NA,(SA_NA,))
         #p2 = pool.apply_async(fcst(NM_hist,NM_param,self.fcst_periods,self.season_periods).exsm_NM,(SM_NM,))
         #p3 = pool.apply_async(fcst(LN_hist,LN_param,self.fcst_periods,self.season_periods).exsm_LN,())
         #p4 = pool.apply_async(fcst(LA_hist,LA_param,self.fcst_periods,self.season_periods).exsm_LA,(SA_LA,))
         #p5 = pool.apply_async(fcst(LM_hist,LM_param,self.fcst_periods,self.season_periods).exsm_LM,(SM_LM,))
         
         NN_fcst = p0.get()
         NA_fcst = p1.get()
         NM_fcst = p2.get()
         LN_fcst = p3.get()
         LA_fcst = p4.get()
         LM_fcst = p5.get()
         
         fcst_idx = np.array(range(1,self.hist.index[-1] + self.fcst_periods + 1))
         fin_fcst = pd.DataFrame(columns=self.hist.columns, index=fcst_idx, data=0)
         
         for i in range(len(fin_fcst.columns)):
              mod = model.iloc[i,0]
              col = fin_fcst.columns[i]
              if len(mod) == 2:
                   fin_fcst[col] = eval(mod + '_fcst')[col]
              else:
                   mod1 = mod[:2]
                   mod2 = mod[3:]
                   
                   fin_fcst[col] = eval(mod1 + '_fcst')[col] * 0.7
                   fin_fcst[col] += eval(mod2 + '_fcst')[col] * 0.3
         
         return fin_fcst
         
         
         