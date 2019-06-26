import pandas as pd
import numpy as np
from pmdarima import auto_arima

class fcst:
    def __init__(self,hist,params=None, breakout=None, fcst_periods=12,season_periods = 12,trend='Additive',seasonality='Additive'):
        self.hist = hist
        self.params = params
        self.breakout = breakout
        self.fcst_periods = fcst_periods
        self.season_periods = season_periods
        self.seasonality = seasonality
        self.trend = trend
        
    def trans_df(self):
        top_df = self.hist.groupby(self.hist.columns[2]).sum().rename(columns={self.hist.columns[3]: 'Total'})
        mid_df = self.hist.groupby([self.hist.columns[0],self.hist.columns[2]], as_index=False).sum()
        mid_cnt = mid_df[self.hist.columns[0]].nunique()
        mid_df = mid_df.pivot(index = self.hist.columns[2], columns = self.hist.columns[0], values = self.hist.columns[3])
        low_df = self.hist.groupby([self.hist.columns[0],self.hist.columns[1],self.hist.columns[2]], as_index=False).sum()
        
        breakout = [mid_cnt]
        for i in range(0,mid_cnt):
            col = mid_df.columns[i]
            low_cnt = low_df[low_df.iloc[:,0]==col].iloc[:,1].nunique()
            breakout.append(low_cnt)

        low_df = low_df.pivot(index = self.hist.columns[2], columns = self.hist.columns[1], values = self.hist.columns[3])
        trans_df = pd.concat([top_df, mid_df, low_df], axis=1, sort=False)
        
        param_df = trans_df[:4].copy().reset_index().drop('Month',axis=1)
        param_df.rename({0:'Alpha',1:'Beta',2:'Gamma',3:'Phi'}, axis='index', inplace=True)
        
        param_df[:1] = 0.7
        param_df[1:2] = 0.5
        param_df[2:3] = 0.6
        param_df[3:4] = 1.0

        ##############
        #need to add replace na/0 logic?
        ##############
        return trans_df, param_df, breakout
    
    def initial_seasonal_components(self):
        seasonals = {}
        season_avg = []
        n_seasons = int(len(self.hist)/self.season_periods)
        for n in range(n_seasons):
            season_avg.append(self.hist[self.season_periods*n:self.season_periods*(n+1)].sum() / self.season_periods)
        for i in range(self.season_periods):
            sum_of_vals_over_avg = 0
            for n in range(n_seasons):
                if self.seasonality == 'Additive':
                    sum_of_vals_over_avg += np.array(self.hist[self.season_periods*n+i:self.season_periods*n+i+1]-np.array(season_avg[n]))
                else: sum_of_vals_over_avg += np.array(self.hist[self.season_periods*n+i:self.season_periods*n+i+1]/np.array(season_avg[n]))
            seasonals[i]= sum_of_vals_over_avg/n_seasons
        return seasonals
    
    def exponential_smoothing(self):
        hist_level = self.hist.copy()
        hist_trend = hist_level.copy()
        forecast = hist_level.copy()
        seasonals = self.initial_seasonal_components()
        
        hist_trend[:] = 0
        if self.trend == 'Additive':
            if self.seasonality != None:
                for i in range(self.season_periods):
                    hist_trend[:1] += (np.array(self.hist[i+self.season_periods:i+self.season_periods+1]) 
                                        -np.array(self.hist[i:i+1])) / self.season_periods
                hist_trend[:1] /= self.season_periods
            else: 
                hist_trend[:1] = np.array(hist_level[1:2]) - hist_level[:1]
        
        for i in range(1,len(hist_level)):
            if self.seasonality == 'Additive':
                hist_level[i:i+1] = (self.params.loc['Alpha']*(self.hist[i:i+1]-np.array(seasonals[i%self.season_periods]))
                                     + np.array(1-self.params.loc['Alpha']) * np.array(hist_level[i-1:i]+hist_trend[i-1:i]))
                
            elif self.seasonality == 'Multiplicative':
                hist_level[i:i+1] = (self.params.loc['Alpha']*(self.hist[i:i+1]/np.array(seasonals[i%self.season_periods]))
                                     + np.array(1-self.params.loc['Alpha']) * np.array(hist_level[i-1:i]+hist_trend[i-1:i]))
            else:
                hist_level[i:i+1] = (self.params.loc['Alpha']*self.hist[i:i+1]
                                     + np.array(1-self.params.loc['Alpha']) * np.array(hist_level[i-1:i]+hist_trend[i-1:i]))
            if self.trend == 'Additive':
                hist_trend[i:i+1] = (self.params.loc['Beta']*(hist_level[i:i+1]-np.array(hist_level[i-1:i]))
                                     + np.array(1-self.params.loc['Beta'])*np.array(hist_trend[i-1:i]))
            if self.seasonality == 'Additive':
                seasonals[i%self.season_periods] = (np.array(self.params.loc['Gamma'])*np.array(self.hist[i:i+1]-hist_level[i:i+1])
                                                    + np.array(1-self.params.loc['Gamma'])*seasonals[i%self.season_periods])
                forecast[i:i+1] = hist_level[i:i+1] + hist_trend[i:i+1] + seasonals[i%self.season_periods]
            elif self.seasonality == 'Multiplicative':
                seasonals[i%self.season_periods] = (np.array(self.params.loc['Gamma'])*np.array(self.hist[i:i+1]/hist_level[i:i+1])
                                                    + np.array(1-self.params.loc['Gamma'])*seasonals[i%self.season_periods])
                forecast[i:i+1] = (hist_level[i:i+1] + hist_trend[i:i+1]) * seasonals[i%self.season_periods]
                
            else:
                seasonals[i%self.season_periods] = 0
                forecast[i:i+1] = (hist_level[i:i+1] + hist_trend[i:i+1])
        
        fin_smooth = np.array(hist_level[-1:])
        fin_trend = np.array(hist_trend[-1:])
        trend = fin_trend
        for i in range(len(hist_level), len(hist_level)+self.fcst_periods):
            forecast = forecast.append(pd.DataFrame(data=hist_level[-1:], index=forecast[-1:].index.shift(1,'MS')))
            if self.seasonality == 'Multiplicative':
                forecast[i:i+1] = (fin_smooth + trend) * seasonals[i%self.season_periods]
            else: 
                forecast[i:i+1] = fin_smooth + trend + seasonals[i%self.season_periods]
            trend = np.array(self.params.loc['Phi']) * trend + fin_trend
        return forecast.fillna(0)
    
    def ex_smooth_tester(self):
        
        fcst_split_1 = fcst(self.hist[:-1],self.params, fcst_periods=1,trend=self.trend,seasonality=self.seasonality).exponential_smoothing()
        fcst_split_2 = fcst(self.hist[:-2],self.params, fcst_periods=2,trend=self.trend,seasonality=self.seasonality).exponential_smoothing()
        fcst_split_3 = fcst(self.hist[:-3],self.params, fcst_periods=3,trend=self.trend,seasonality=self.seasonality).exponential_smoothing()
        fcst_split_4 = fcst(self.hist[:-4],self.params, fcst_periods=4,trend=self.trend,seasonality=self.seasonality).exponential_smoothing()
        return [fcst_split_1,fcst_split_2,fcst_split_3,fcst_split_4]
    
    def ARIMA(self, history):
        stepwise_model = auto_arima(history, start_p=1, start_q=1, max_p=3, max_q=3, m=12,
                        start_P=1, seasonal=True, error_action='ignore',
                        suppress_warnings=True, stepwise=True)
        in_sample = stepwise_model.predict_in_sample()
        forecast = stepwise_model.fit_predict(history,n_periods=self.fcst_periods)
        ARIMA_fcst = np.append(in_sample,forecast)
        return ARIMA_fcst
    
    def fcst_tester(self,history,forecasts,blend=False):
        hist = history.copy()
        fcst_split_1 = forecasts[0]
        fcst_split_2 = forecasts[1]
        fcst_split_3 = forecasts[2]
        fcst_split_4 = forecasts[3]
        
        m1_APE_1 = abs(fcst_split_1[-1:]-hist[-1:])/hist[-1:]
        m1_APE_2 = abs(fcst_split_2[-2:-1]-hist[-2:-1])/hist[-2:-1]
        m1_APE_3 = abs(fcst_split_3[-3:-2]-hist[-3:-2])/hist[-3:-2]
        m1_APE_4 = abs(fcst_split_4[-4:-3]-hist[-4:-3])/hist[-4:-3]
        
        m2_APE_2 = abs(fcst_split_2[-1:]-hist[-1:])/hist[-1:]
        m2_APE_3 = abs(fcst_split_3[-2:-1]-hist[-2:-1])/hist[-2:-1]
        m2_APE_4 = abs(fcst_split_4[-3:-2]-hist[-3:-2])/hist[-3:-2]
        
        m3_APE_3 = abs(fcst_split_3[-1:]-hist[-1:])/hist[-1:]
        m3_APE_4 = abs(fcst_split_4[-2:-1]-hist[-2:-1])/hist[-2:-1]
        
        m1_MAPE = m1_APE_1.sum()*0.5 + m1_APE_2.sum()*0.25 + m1_APE_3.sum()*0.15 + m1_APE_4.sum()*0.1
        m2_MAPE = m2_APE_2.sum()*0.6 + m2_APE_3.sum()*0.3 + m2_APE_4.sum()*0.1
        m3_MAPE = m3_APE_3.sum()*0.7 + m3_APE_4.sum()*0.3
        
        if blend == True:
            #MAPE = (q1_MAPE*0.6 + q2_MAPE*0.3 + q3_MAPE*0.1)
            MAPE = (m1_MAPE*0.6 + m2_MAPE*0.3 + m3_MAPE*0.1)
        else: 
            #MAPE = (q1_MAPE, q2_MAPE, q3_MAPE)
            MAPE = (m1_MAPE, m2_MAPE, m3_MAPE)
        return MAPE
    
    def fcst_optimizer(self):
        top_fcsts = {}
        top_MAPE = pd.DataFrame(columns=self.hist.columns,data=999,index=['MAPE'])
        top_param = self.params.copy()
        top_param[0:3] = 0.5
        param_adj = 1
        
        for level in range(0,5):
            level_top = top_param.copy()
            param_adj /= 2
            for a in range(-1,2):
                param_opt = level_top.copy()
                param_opt.loc['Alpha'] = np.array(level_top.loc['Alpha']) + a * param_adj
                
                if self.trend == 'Additive':
                    for b in range(-1,2):
                        param_opt.loc['Beta'] = np.array(level_top.loc['Beta']) + b * param_adj
                        
                        if self.seasonality != None:
                            for g in range(-1,2):
                                param_opt.loc['Gamma'] = np.array(level_top.loc['Gamma']) + g * param_adj
                                param_opt[param_opt > 1] = 1
                                param_opt[param_opt < 0] = 0
                                forecasts = fcst(self.hist,param_opt,trend=self.trend,seasonality=self.seasonality).ex_smooth_tester()
                                MAPE = fcst(self.hist,param_opt,trend=self.trend,seasonality=self.seasonality).fcst_tester(self.hist,forecasts)
                                Q1_MAPE = MAPE[0]
                                for column in range(len(top_MAPE.columns)):
                                    if float(top_MAPE.iloc[:,column]) > np.array(Q1_MAPE)[column]:
                                        top_MAPE.iloc[:,column] = np.array(Q1_MAPE)[column]
                                        top_param.iloc[0:3,column] = param_opt.iloc[0:3,column]
                                        top_fcsts[column] = [forecasts[0].iloc[:,column],forecasts[1].iloc[:,column],forecasts[2].iloc[:,column],forecasts[3].iloc[:,column]]
                                
                        else:
                            param_opt.loc['Gamma'] = 1
                            forecasts = fcst(self.hist,param_opt,trend=self.trend,seasonality=self.seasonality).ex_smooth_tester()
                            MAPE = fcst(self.hist,param_opt,trend=self.trend,seasonality=self.seasonality).fcst_tester(self.hist,forecasts)
                            Q1_MAPE = MAPE[0]
                            for column in range(len(top_MAPE.columns)):
                                if float(top_MAPE.iloc[:,column]) > np.array(Q1_MAPE)[column]:
                                    top_MAPE.iloc[:,column] = np.array(Q1_MAPE)[column]
                                    top_param.iloc[0:3,column] = param_opt.iloc[0:3,column]
                                    top_fcsts[column] = [forecasts[0].iloc[:,column],forecasts[1].iloc[:,column],forecasts[2].iloc[:,column],forecasts[3].iloc[:,column]]

                else:
                    param_opt.loc['Beta'] = 1
                    if self.seasonality != None:
                        for g in range(-1,2):
                            param_opt.loc['Gamma'] = np.array(level_top.loc['Gamma']) + g * param_adj
                            param_opt[param_opt > 1] = 1
                            param_opt[param_opt < 0] = 0
                            forecasts = fcst(self.hist,param_opt,trend=self.trend,seasonality=self.seasonality).ex_smooth_tester()
                            MAPE = fcst(self.hist,param_opt,trend=self.trend,seasonality=self.seasonality).fcst_tester(self.hist,forecasts)
                            Q1_MAPE = MAPE[0]
                            for column in range(len(top_MAPE.columns)):
                                if float(top_MAPE.iloc[:,column]) > np.array(Q1_MAPE)[column]:
                                    top_MAPE.iloc[:,column] = np.array(Q1_MAPE)[column]
                                    top_param.iloc[0:3,column] = param_opt.iloc[0:3,column]
                                    top_fcsts[column] = [forecasts[0].iloc[:,column],forecasts[1].iloc[:,column],forecasts[2].iloc[:,column],forecasts[3].iloc[:,column]]
                                
                    else:
                        param_opt.loc['Gamma'] = 1
                        forecasts = fcst(self.hist,param_opt,trend=self.trend,seasonality=self.seasonality).ex_smooth_tester()
                        MAPE = fcst(self.hist,param_opt,trend=self.trend,seasonality=self.seasonality).fcst_tester(self.hist,forecasts)
                        Q1_MAPE = MAPE[0]
                        for column in range(len(top_MAPE.columns)):
                            if float(top_MAPE.iloc[:,column]) > np.array(Q1_MAPE)[column]:
                                top_MAPE.iloc[:,column] = np.array(Q1_MAPE)[column]
                                top_param.iloc[0:3,column] = param_opt.iloc[0:3,column]
                                top_fcsts[column] = [forecasts[0].iloc[:,column],forecasts[1].iloc[:,column],forecasts[2].iloc[:,column],forecasts[3].iloc[:,column]]
                                
        if self.trend == 'Additive':
            top_phi = pd.DataFrame(columns=self.hist.columns,data=999,index=['MAPE'])
            top_param[3:] = 0.5
            param_adj = 1
            for phi_level in range(0,5):
                level_top = top_param.copy()
                param_adj /= 2
                
                for p in range(-1,2):
                    param_opt = level_top.copy()
                    param_opt.loc['Phi'] = np.array(level_top.loc['Phi']) + p * param_adj
                    param_opt[param_opt > 1] = 1
                    param_opt[param_opt < 0] = 0
                    forecasts = fcst(self.hist,param_opt,trend=self.trend,seasonality=self.seasonality).ex_smooth_tester()
                    MAPE = fcst(self.hist,param_opt,trend=self.trend,seasonality=self.seasonality).fcst_tester(self.hist,forecasts,blend=True)
                    for column in range(len(top_phi.columns)):
                        if float(top_phi.iloc[:,column]) > np.array(MAPE)[column]:
                            top_phi.iloc[:,column] = np.array(MAPE)[column]
                            top_param.iloc[3:,column] = param_opt.iloc[3:,column]
                            top_fcsts[column] = [forecasts[0].iloc[:,column],forecasts[1].iloc[:,column],forecasts[2].iloc[:,column],forecasts[3].iloc[:,column]]
                            
            top_MAPE = top_phi
            
        else:
            forecasts = fcst(self.hist,param_opt,trend=self.trend,seasonality=self.seasonality).ex_smooth_tester()
            MAPE = fcst(self.hist,param_opt,trend=self.trend,seasonality=self.seasonality).fcst_tester(self.hist,forecasts,blend=True)
            top_MAPE = pd.DataFrame(columns=self.hist.columns,data=np.reshape(np.array(MAPE),(1,len(MAPE))))
            
        return top_param, top_MAPE, top_fcsts
    
    def fcst_rank(self,NN_list,NA_list,NM_list,AN_list,AA_list,AM_list,AR_list):
        #Run all forecasts, pick the best
        
        NN_par, NN_MAPE, NN_fcst = NN_list
        NA_par, NA_MAPE, NA_fcst = NA_list
        NM_par, NM_MAPE, NM_fcst = NM_list
        AN_par, AN_MAPE, AN_fcst = AN_list
        AA_par, AA_MAPE, AA_fcst = AA_list
        AM_par, AM_MAPE, AM_fcst = AM_list
        AR_MAPE, AR_fcst = AR_list
        
        #NN_par, NN_MAPE, NN_fcst = fcst(self.hist,self.params,fcst_periods=12,trend=None,seasonality=None).fcst_optimizer()
        #NA_par, NA_MAPE, NA_fcst = fcst(self.hist,self.params,fcst_periods=12,trend=None,seasonality='Additive').fcst_optimizer()
        #NM_par, NM_MAPE, NM_fcst = fcst(self.hist,self.params,fcst_periods=12,trend=None,seasonality='Multiplicative').fcst_optimizer()
        #AN_par, AN_MAPE, AN_fcst = fcst(self.hist,self.params,fcst_periods=12,trend='Additive',seasonality=None).fcst_optimizer()
        #AA_par, AA_MAPE, AA_fcst = fcst(self.hist,self.params,fcst_periods=12,trend='Additive',seasonality='Additive').fcst_optimizer()
        #AM_par, AM_MAPE, AM_fcst = fcst(self.hist,self.params,fcst_periods=12,trend='Additive',seasonality='Multiplicative').fcst_optimizer()
        
        NN_par.index=['NN_Alpha','NN_Beta','NN_Gamma','NN_Phi']
        NA_par.index=['NA_Alpha','NA_Beta','NA_Gamma','NA_Phi']
        NM_par.index=['NM_Alpha','NM_Beta','NM_Gamma','NM_Phi']
        AN_par.index=['AN_Alpha','AN_Beta','AN_Gamma','AN_Phi']
        AA_par.index=['AA_Alpha','AA_Beta','AA_Gamma','AA_Phi']
        AM_par.index=['AM_Alpha','AM_Beta','AM_Gamma','AM_Phi']
        
        opt_params = NN_par.append(NA_par.append(NM_par.append(AN_par.append(AA_par.append(AM_par)))))
        
        NN_MAPE.index=['NN']
        NA_MAPE.index=['NA']
        NM_MAPE.index=['NM']
        AN_MAPE.index=['AN']
        AA_MAPE.index=['AA']
        AM_MAPE.index=['AM']
        #AR_MAPE.index=['AR']
        
        MAPE = NN_MAPE.append(NA_MAPE.append(NM_MAPE.append(AN_MAPE.append(AA_MAPE.append(AM_MAPE.append(AR_MAPE))))))
        #MAPE = NN_MAPE.append(NA_MAPE.append(NM_MAPE))
        #MAPE = NN_MAPE.append(NA_MAPE)
        blend_MAPE = []
        blend_model = []
        blend_fcsts = {}
        
        for column in range(len(MAPE.columns)):
            M1 = MAPE.iloc[:,column].nsmallest(2).index[0]
            M2 = MAPE.iloc[:,column].nsmallest(2).index[1]
            blend_model.append(M1+'_'+M2)
            blend_fcst = [sum(x)/2 for x in zip(eval(M1+'_fcst')[column],eval(M2+'_fcst')[column])]
            blend_fcsts[column] = blend_fcst
            new_MAPE = fcst(self.hist).fcst_tester(self.hist.iloc[:,column],blend_fcst,blend=True)
            blend_MAPE.append(float(new_MAPE))

        MAPE.loc['blend'] = blend_MAPE
        
        best_MAPE = MAPE.min()
        best_model = np.array(MAPE.idxmin())
        best_fcsts = {}
        
        for idx, model in enumerate(best_model):
            if model == 'blend':
                best_fcsts[idx] = blend_fcsts[idx]
                best_model[idx] = blend_model[idx]
            else:
                best_fcsts[idx] = eval(model+'_fcst')[idx]
        
        return best_model, best_MAPE, best_fcsts, opt_params, MAPE #MAPE, blend_model
    
    def allocation(self, opt_vars):
        
        top_models, top_mapes, top_fcsts, opt_params = opt_vars #fcst(self.hist,self.params,fcst_periods=12).fcst_rank()
        
        col_index = self.breakout[0] + 1
        mid_tot_cache = [] #added here, loop issue
        mid_mape_cache = [] #added here, loop issue
        for i in range(0,self.breakout[0]): #For each mid
            
            mid_tot = [0,0,0,0]
            
            for key,value in top_fcsts.items(): #look at each row
                if key in range(col_index,col_index+self.breakout[i+1]): #look at lows in range of the mid
                    mid_tot = [sum(x) for x in zip(mid_tot, value)] #sum each of 4 forecasts for each low in range
                    
            mid_tot_cache.append(mid_tot)
            mid_mape = fcst(self.hist).fcst_tester(self.hist.iloc[:,i+1],mid_tot,blend=True)
            mid_mape_cache.append(mid_mape)
            
            #turn off mid agg!!! XXXXXXXX
            #if mid_mape < top_mapes[i+1]:
            #    top_models[i+1] = 'AG_low'
            #    top_fcsts[i+1] = mid_tot
            #    top_mapes[i+1] = mid_mape
            
            col_index += self.breakout[i+1]
            
        tot_mid = [0,0,0,0]
        tot_low = [0,0,0,0]
        for key,value in top_fcsts.items():
            if key == 0: next
            elif key in range(1,self.breakout[0]+1): tot_mid = [sum(x) for x in zip(tot_mid, value)]
            else: tot_low = [sum(x) for x in zip(tot_low, value)]
                
        top_mape = fcst(self.hist).fcst_tester(self.hist.iloc[:,0],top_fcsts[0],blend=True)
        mid_mape = fcst(self.hist).fcst_tester(self.hist.iloc[:,0],tot_mid,blend=True)
        low_mape = fcst(self.hist).fcst_tester(self.hist.iloc[:,0],tot_low,blend=True)
        
        print(top_mape)
        print(mid_mape)
        print(low_mape)
        
        if top_mape <= mid_mape and top_mape <= low_mape: next
        elif mid_mape < top_mape and mid_mape < low_mape:
            print('mid')
            top_models[0] = 'AG_mid'
            top_fcsts[0] = tot_mid
            top_mapes[0] = mid_mape
        else:
            print('low')
            top_models[0] = 'AG_low'
            top_fcsts[0] = tot_low
            top_mapes[0] = low_mape
        
        top_models[0] = 'AG_mid' #force AG_mid XXXXXX
        
        if top_models[0][:2] == 'AG':
            if top_models[0][3:] == 'mid':
                col_index = self.breakout[0] + 1
                for i in range(0,self.breakout[0]):
                    if top_models[i+1][:2] == 'AG': next
                    else:
                        mid_tot = top_fcsts[i+1]
                        mid_cache = mid_tot_cache[i]
                        
                        for j in range(col_index, col_index + self.breakout[i+1]):
                            top_models[j] = 'TD_'+top_models[j]
                            fcst_perc = [x / y for x, y in zip(top_fcsts[j], mid_cache)]
                            top_fcsts[j] = [x * y for x, y in zip(fcst_perc, mid_tot)]
                            top_mapes[j] = fcst(self.hist).fcst_tester(self.hist.iloc[:,j],top_fcsts[j],blend=True)
                            
                    col_index += self.breakout[i+1]
            else:
                col_index = self.breakout[0] + 1
                for i in range(0,self.breakout[0]):
                    if top_models[i+1][:2] == 'AG': next
                    else:
                        top_models[i+1] = 'AG_low'
                        top_fcsts[i+1] = mid_tot_cache[i]
                        top_mapes[i+1] = mid_mape_cache[i]
        else:
            print('top')
            mid_agg = [0,0,0,0]
            for key,value in top_fcsts.items():
                if key in range(1,self.breakout[0]+1):
                    mid_agg = [sum(x) for x in zip(mid_agg, value)]
            
            col_index = self.breakout[0] + 1
            for i in range(0,self.breakout[0]):
                top_models[i+1] = 'TD_'+top_models[i+1]
                fcst_perc = [x / y for x, y in zip(top_fcsts[i+1],mid_agg)]
                top_fcsts[i+1] = [x * y for x, y in zip(fcst_perc, top_fcsts[0])]
                top_mapes[i+1] = fcst(self.hist).fcst_tester(self.hist.iloc[:,i+1],top_fcsts[i+1],blend=True)
                
                for j in range(col_index, col_index + self.breakout[i+1]):
                    top_models[j] = 'TD_'+top_models[j]
                    low_perc = [x / y for x, y in zip(top_fcsts[j], mid_tot_cache[i])]
                    top_fcsts[j] = [x * y for x, y in zip(low_perc, top_fcsts[i+1])]
                    top_mapes[i+1] = fcst(self.hist).fcst_tester(self.hist.iloc[:,j],top_fcsts[j],blend=True)
                col_index += self.breakout[i+1]
            
        return top_models, top_mapes, top_fcsts, opt_params
    
    def top_fcst(self, opt_vars):
        
        top_mod, fcst_MAPE, legacy_fcst, opt_params = fcst(self.hist,self.params,breakout=self.breakout).allocation(opt_vars)
        
        NN_hist = self.hist.copy()
        NA_hist = self.hist.copy()
        NM_hist = self.hist.copy()
        AN_hist = self.hist.copy()
        AA_hist = self.hist.copy()
        AM_hist = self.hist.copy()
        AR_hist = self.hist.copy()
        
        NN_pars = opt_params[:4]
        NA_pars = opt_params[4:8]
        NM_pars = opt_params[8:12]
        AN_pars = opt_params[12:16]
        AA_pars = opt_params[16:20]
        AM_pars = opt_params[20:24]
        
        NN_pars.index = ['Alpha','Beta','Gamma','Phi']
        NA_pars.index = ['Alpha','Beta','Gamma','Phi']
        NM_pars.index = ['Alpha','Beta','Gamma','Phi']
        AN_pars.index = ['Alpha','Beta','Gamma','Phi']
        AA_pars.index = ['Alpha','Beta','Gamma','Phi']
        AM_pars.index = ['Alpha','Beta','Gamma','Phi']
        
        fin_fcst = fcst(NN_hist,NN_pars,fcst_periods=12,trend=None,seasonality=None).exponential_smoothing() * 0
        index_df = self.hist[:1].copy() * 0 + 7
        index = -1
        
        
        for model in top_mod:
            index += 1
            if model.find('NN') == -1:
                NN_hist.iloc[:,index] *= 0
                index_df.iloc[:,index] -= 1
            if model.find('NA') == -1:
                NA_hist.iloc[:,index] *= 0
                index_df.iloc[:,index] -= 1
            if model.find('NM') == -1:
                NM_hist.iloc[:,index] *= 0
                index_df.iloc[:,index] -= 1
            if model.find('AN') == -1:
                AN_hist.iloc[:,index] *= 0
                index_df.iloc[:,index] -= 1
            if model.find('AA') == -1:
                AA_hist.iloc[:,index] *= 0
                index_df.iloc[:,index] -= 1
            if model.find('AM') == -1:
                AM_hist.iloc[:,index] *= 0
                index_df.iloc[:,index] -= 1
            if model.find('AR') == -1:
                fin_fcst.iloc[:,index] *= 0
                index_df.iloc[:,index] -= 1
            else:
                fin_fcst.iloc[:,index] = fcst(self.hist,fcst_periods=12).ARIMA(AR_hist.iloc[:,index])
                
        fin_fcst += fcst(NN_hist,NN_pars,fcst_periods=12,trend=None,seasonality=None).exponential_smoothing()
        fin_fcst += fcst(NA_hist,NA_pars,fcst_periods=12,trend=None,seasonality='Additive').exponential_smoothing()
        fin_fcst += fcst(NM_hist,NM_pars,fcst_periods=12,trend=None,seasonality='Multiplicative').exponential_smoothing()
        fin_fcst += fcst(AN_hist,AN_pars,fcst_periods=12,trend='Additive',seasonality=None).exponential_smoothing()
        fin_fcst += fcst(AA_hist,AA_pars,fcst_periods=12,trend='Additive',seasonality='Additive').exponential_smoothing()
        fin_fcst += fcst(AM_hist,AM_pars,fcst_periods=12,trend='Additive',seasonality='Multiplicative').exponential_smoothing()
        
        for column in index_df:
            if float(index_df[column]) > 1:
                fin_fcst[column] /= float(index_df[column])

        #Allocation: Adds up AG mids first
        col_index = self.breakout[0] + 1
        for i in range(0,self.breakout[0]):
            
            mid_agg = [0]
            mid_tot = [0]
            
            if top_mod[i+1].find('AG') > -1:
                for low in range(col_index, col_index+self.breakout[i+1]):
                    mid_tot += fin_fcst.iloc[:,low]
                fin_fcst.iloc[:,i+1] = mid_tot
            
            mid_agg += fin_fcst.iloc[:,i+1]
            col_index += self.breakout[i+1]
            
        #top-mid aggrigation
        col_index = self.breakout[0] + 1
        top_agg = [0]
        for i in range(0,self.breakout[0]):
            if top_mod[0] == 'AG_mid':
                top_agg += fin_fcst.iloc[:,i+1]
                if top_mod[i+1].find('AG') == -1:
                    low_agg = [0]
                    for low in range(col_index, col_index+self.breakout[i+1]):
                        low_agg += fin_fcst.iloc[:,low]
                    for low in range(col_index, col_index+self.breakout[i+1]):
                        low_perc = fin_fcst.iloc[:,low] / low_agg
                        fin_fcst.iloc[:,low] = fin_fcst.iloc[:,i+1] * low_perc

            if top_mod[0] == 'AG_low':
                mid_agg = [0]
                for low in range(col_index, col_index+self.breakout[i+1]):
                    mid_agg += fin_fcst.iloc[:,low]
                fin_fcst.iloc[:,i+1] = mid_agg
                top_agg += mid_agg
                
            col_index += self.breakout[i+1]
        
        if top_mod[0][:2] == 'AG':
            fin_fcst.iloc[:,0] = top_agg
        else:
            mid_tot = [0]
            for i in range(0,self.breakout[0]):
                mid_tot += fin_fcst.iloc[:,i+1]
                
            col_index = self.breakout[0] + 1
            for i in range(0,self.breakout[0]):
                mid_perc = fin_fcst.iloc[:,i+1] / mid_tot
                fin_fcst.iloc[:,i+1] = fin_fcst.iloc[:,0] * mid_perc
                
                low_agg = [0]
                for low in range(col_index, col_index+self.breakout[i+1]):
                    low_agg += fin_fcst.iloc[:,low]
                for low in range(col_index, col_index+self.breakout[i+1]):
                    low_perc = fin_fcst.iloc[:,low] / low_agg
                    fin_fcst.iloc[:,low] = fin_fcst.iloc[:,i+1] * low_perc
                    
                col_index = self.breakout[0] + 1
            
        return top_mod, fcst_MAPE, fin_fcst, opt_params
    