####################################
# Module for evaluation of diffusivity measurements
# Authors: Stefan Strohauer, Noah Ploch, Fabian Wietschorke
# 2019 - 2021
####################################

## R_NC muss noch in
# maybe check meaningfulness of error calculation of T_Bc2
# check for "is false"

from import_lib import *
import scipy
import scipy.constants as constants
#import json as js

class DiffusivityMeasurement:
    '''Input: filename with raw data, (fit function type)
    Output: calculation of diffusivity and related values that are stored and/or returned by executed methods
    Description: This class contains all properties concerning the preparation of the measurement data. Also,
    its methods control the workflow to determine the electron diffusivity.'''

    def __init__(self, filename, takeAbsB=True, key_w_data=None, fit_function='R_max/2', T_sweeps=None, T_selector = "T_sample",invertR=False,bin_size=0.01): # to see why it is cleaner to create new object
                                  # https://stackoverflow.com/questions/29947810/reusing-instances-of-objects-vs-creating-new-ones-with-every-update
        self._filename = str(filename)
        self._key_w_data = key_w_data  # for old measurement schemes: data, data_old

        # here measurement data is stored in a structured manner
        self._time_sweeps = []
        self._T_sample_sweeps = []
        self._T_PCB_sweeps = []
        self._T_sample_ohms_sw = []
        self._T_PCB_ohms_sw = []
        self._B_sweeps = []
        self._R_sweeps = []
        self._raw_data = self.__import_file(key_w_data)


        if takeAbsB==True:
            self._B_sweeps = [[abs(elem) for elem in sublist] for sublist in self._B_sweeps]  #a function that loops over all B values, takes the absolute value and overwrites the old one

        if invertR==True:
            self._R_sweeps = [[-elem for elem in sublist] for sublist in self._R_sweeps]  #a function that loops over all R values, inverts the value and overwrites the old one

        self.__RT_sweeps_per_B = {}  # sorting temperature and resistance after magnetic field (B: {T,R})
        self._parameters_RTfit = {}  # dictionary of B fields with corresponding fit parameters from fitting {T,R}   {B:{fit parameters}}
        self._fitted_RTvalues = {}  # fitted values of Resistance R for given Temperature T at magnetic fields B
        self.__RT_fit_low_lim = 0  # lower limit of RT fits
        self.__RT_fit_upp_lim = np.inf  # upper limit of RT fits

        self._diffusivity = 0
        self._diffusivity_err = 0
        self._Tc_0T = 0   #Where is this still in use?
        self._T_selector = T_selector # options: T_sample or T_PCB
        self.default_B_array = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        self._B_bin_size = bin_size #Default value is 0.01, can be changed when creating new object
        self.sheet_resistance_geom_factor = 4.53 # geometry factor to determine sheet resistance 4.53=pi/ln(2)
                                                 # calculated sheet resistance of the film
        self.__rearrange_B_sweeps(T_sweeps)
        self.max_measured_resistance = self.get_R_NC()
#        self.sheet_resistance = self.sheet_resistance_geom_factor*self.max_measured_resistance

        self._RTfit = RTfit(fit_function)  # initializing RT fit class
        self.Bc2vsT_fit = None

    @property
    def filename(self):
        return self._filename

    @property
    def key_w_data(self):
        return self._key_w_data

    @property
    def time_sweeps(self):
        return self._time_sweeps

    @property
    def T_sample_sweeps(self):
        return self._T_sample_sweeps

    @property
    def T_PCB_sweeps(self):
        return self._T_PCB_sweeps

    @property
    def T_sample_ohms_sw(self):
        return self._T_sample_ohms_sw

    @property
    def B_sweeps(self):
        return self._B_sweeps

    @property
    def R_sweeps(self):
        return self._R_sweeps

    @property
    def raw_data(self):
        return self._raw_data

    @property
    def parameters_RTfit(self):
        return self._parameters_RTfit

    @property
    def fitted_RTvalues(self):
        return self._fitted_RTvalues

    @property
    def diffusivity(self):
        return self._diffusivity

    @property
    def diffusivity_err(self):
        return self._diffusivity_err

    @property
    def Tc_0T(self):
        return self._Tc_0T

    @property
    def T_selector(self):
        return self._T_selector

    @property
    def B_bin_size(self):
        return self._B_bin_size

    @property
    def RTfit(self):
        return self._RTfit


    def __import_file(self, key_with_data=None):
        """Input: data key with measurement data, filename (from properties)
        Output: structured measurement data
        Description: reads string with filename and returns measurement data from json and .mat files. Also, reads into the
        corresponding attributes the measurement data. Differentiates between json and mat file through setting the key_with_data parameter"""
        if set('.json').issubset(set(self._filename)):
            with open(self._filename) as json_data:
                data_raw = js.load(json_data)
                self._time_sweeps, self._T_sample_sweeps, self._T_PCB_sweeps, self._T_sample_ohms_sw, self._T_PCB_ohms_sw, self._B_sweeps, self._R_sweeps = \
                    self.__return_measurement_data(data_raw['sweep'], 't', 'T_sample', 'T_PCB', 'T_sample_ohms', 'T_PCB_ohms', 'B', 'R')
        elif set('.mat').issubset(set(self._filename)):
            data = io.loadmat(self._filename, squeeze_me=True)
            if key_with_data in list(data.keys()):  # this try except structure tries to look up the data through the given dict key and, if it is None or not valid (none is not valid), tries the dict keys we usually used in the past measurements.
                data_raw = data[key_with_data]
            elif ('data' in data.keys()) or ('data_old' in data.keys()):
                key_list = list(data.keys())
                self._key_w_data = list(filter(lambda x:'data' in x, key_list))[0] # filters from the keys in the data dictionary if it is data or data_old
                data_raw = data[self._key_w_data]
            else:
                raise Exception('Unknown measurement. Data was not found in .mat structure')
            self._time_sweeps, self._B_sweeps, self._R_sweeps = self.__return_measurement_data(data_raw, 8, 9, 10)
            self._T_sample_sweeps = self.__time_temperature_mapping(data_raw)
        else: raise Exception('wrong data type: Please check if correct parameters were passed')

        return data_raw

    def __return_measurement_data(self, data, *args):
        """Input: raw measurement data, indices/keys of measurment quantities dict in raw data
        Output: list of measurement quantites to be unpacked
        Description: reads the different measurement quantities from the raw data into a list that can be unpacked""" #index_t=8, index_B=9, index_R=10
        data_to_properties = []
        for arg in args:
            data_to_properties.append([i[arg] for i in data])
        return data_to_properties

    def __rearrange_B_sweeps(self, T_values = None):
        """Input: T-sweeps, B-sweeps, R-sweeps
        Output: dictionary {B-field: {T,R}}
        Description: rearranges the raw data saved in B-sweeps into RT-sweeps"""
        if self._T_selector == "T_sample":
            T_sweeps = Tools.selector(T_values, self._T_sample_sweeps)
        elif self._T_selector == "T_PCB":
            T_sweeps = Tools.selector(T_values, self._T_PCB_sweeps)
        elif T_values != None:  # maybe these 2 lines can be deleted as this
            T_sweeps = T_values # case is already coverd above with Tools.selector
        else:
            raise ValueError('Only T_sample and T_PCB are valid options for T_selector.')

        TBR_dict = {'T': self.__flatten_list(T_sweeps),
                    'B': self.__flatten_list(self._B_sweeps),
                    'R': self.__flatten_list(self._R_sweeps)}
        TBR_dict['B'] = self.__round_to_decimal(TBR_dict['B'], base=self._B_bin_size)
        for t, b, r in zip (TBR_dict['T'], TBR_dict['B'], TBR_dict['R']):
            try:
                self.__RT_sweeps_per_B[b].append([t,r])
            except:
                self.__RT_sweeps_per_B[b] = [[t,r]]
        for key, value in self.__RT_sweeps_per_B.items():
            value=np.asarray(value)
            self.__RT_sweeps_per_B[key] = {'T': value[:,0], 'R':value[:,1]}

    def __flatten_list(self, array):
        return np.concatenate(array).ravel()

    def __round_to_decimal(self, x, base):

      """Input: scalar or array
      Output: scalar or array rounded to arbitrary decimal number, e.g. 0.3
      Description: rounds to arbitrary decimal number. Does so by multiplying the decimal to round with
      the number of times the decimal fits into the scalar/array. In case the scalar/array has less decimals than
      intended to round, a further rounding is performed as outer function."""
      prec = str(base)[::-1].find('.')
      return np.round(base * np.round(x/base), prec)

    def __time_temperature_mapping(self, data):
        '''Input: temperature data
        Output: interpolated temperature data with same length as R and B-sweeps
        Description: For measurements without Cernox. As temperature is measured between B-sweeps, the time array is used
        to interpolate the temperature (temperature is assumed to rise almost linearly).'''
        # the old data structure is: data =
        # prev_temp_diode, curr_temp_diode, prev_temp_sample, curr_temp_sample,
        # prev_volt_diode, curr_volt_diode, prev_volt_sample, curr_volt_sample,
        # curr_time, curr_mag_field, curr_resistance
        Temp_at_timestamp = [i[0:4] for i in data]
        T_array = []
        for i,j in zip(self._time_sweeps, Temp_at_timestamp):
            t = [i[0], i[-1]]  # locations of the time and temperature data
            T = [j[2], j[3]] # indices 2, 3 yield temperature of the sample diode
            p = np.polyfit(t,T,1)
            T_array.append(np.polyval(p,i))
        return T_array

    def R_vs_T(self, B=None, err=False):
        '''Input: B fields for which RvsT should be returned, flag to decide if error is returned
        Output: Tuple of varying size depending on err-flag including (T,R,T_err,R_err) of an RT-sweep (Resistance-Temperature-Sweep)
        Description: Depending on the input B fields "B" and the err-flag, this method returns the associated RT-sweep with error'''
        B = Tools.selector(B, self.default_B_array, np.array(list(self.__RT_sweeps_per_B.keys())))
        B = self.__round_to_decimal(np.array(B), base=self._B_bin_size)
        if isinstance(B, (int, float)) and not(err):
            return (self.__RT_array_builder(B, 'T'), self.__RT_array_builder(B, 'R'))
        elif isinstance(B, (int, float)) and err:
            T_err = self.__RT_array_builder(B, 'T', self._RTfit.T_meas_error_pc, self._RTfit.T_meas_error_std_dev)
            R_err = self.__RT_array_builder(B, 'R', self._RTfit.R_meas_error_pc)
            return (self.__RT_array_builder(B, 'T'), self.__RT_array_builder(B, 'R'), T_err, R_err)
        elif isinstance(B, (np.ndarray, list)) and not(err):
            return (self.__RT_dict_builder(B, 'T'), self.__RT_dict_builder(B, 'R'))
        elif isinstance(B, (np.ndarray, list)) and err:
            T_dict_err = self.__RT_dict_builder(B, 'T', self._RTfit.T_meas_error_pc, self._RTfit.T_meas_error_std_dev)
            R_dict_err = self.__RT_dict_builder(B, 'R', self._RTfit.R_meas_error_pc)
            return (self.__RT_dict_builder(B, 'T'), self.__RT_dict_builder(B, 'R'), T_dict_err, R_dict_err)
        else: raise TypeError('some input parameter has not a valid type. check input parameters')

    def __RT_array_builder(self, B, selector, error_pc=1, error_std=0):  # Method to return either the R (T) array or the R_err (T_err) array
        return self.__RT_sweeps_per_B[B][selector] * error_pc + error_std

    def __RT_dict_builder(self, B, selector, error_pc=1, error_std=0):  # Method returning either R(R_err)/T(T_err) arrays in form of dicts (B as key)
        return {key:value[selector] * error_pc + error_std for key, value in self.__RT_sweeps_per_B.items() if key in B}

    def get_R_NC(self, upp_lim=np.Inf,return_error=False):
        '''Input: upper limit for measured resistance
        Output: maximum measured resistance (of all sweeps) within limits
        Description: Determine maximum measured resistance (usually close to 15K). Therefore all
        R values with a temperature above 19K are collected and their average is returned as Resistance
        of the normalconduting state.'''
        # max_R_NC = 0
        #
        # for value in self.__RT_sweeps_per_B.values():
        #     curr_R = np.max(value['R'])
        #     if curr_R > max_R_NC and curr_R <= upp_lim:
        #          max_R_NC = curr_R
        #
        # self.max_measured_resistance = max_R_NC
        # self.sheet_resistance = max_R_NC*self.sheet_resistance_geom_factor
        # return max_R_NC

        ## Take all R values above 19 K Temperature. Then take the average to get a more acurrate R_20K.
        T,R=self.R_vs_T(B='all')
        R_above_19K=np.array([])
        for key,t, r in zip(T.keys(), T.values(), R.values()):
            R_above_19K=np.concatenate((R_above_19K,r[t>19]))
        if return_error:
            # uncertainty=max(R_above_19K)-np.average(R_above_19K)  #option 1: distance to maximum
            uncertainty=np.std(R_above_19K)         #option 2: standard deviation of values
            return (np.average(R_above_19K),uncertainty)
        else:
            return(np.average(R_above_19K))

    def R_vs_B(self, B_sweep_index, err=False):
        '''Input: index of the searched B sweep in the data
        Output: R-B-sweep as measured, optionally with error
        Description: Returns R-B-sweeps'''
        # TODO: add option, if no index is provided,all sweeps are returned
        if err:
            B_err = self._B_sweeps[B_sweep_index] * self.Bc2vsT_fit.B_field_meas_error_pc + self.Bc2vsT_fit.B_field_meas_error_round
            R_err = self._R_sweeps[B_sweep_index] * self._RTfit.R_meas_error_pc
            return (self._B_sweeps[B_sweep_index], self._R_sweeps[B_sweep_index], B_err, R_err)
        else: return (self._B_sweeps[B_sweep_index], self._R_sweeps[B_sweep_index])

    def Bc2_vs_T(self, err=False):
        '''Input: error flag
        Output: Bc2(T) as arrays (T,Bc2), (optional) with error
        Description: returns Bc2(T). Data comes from the Bc2vsTfit class initialied in calc_diffusivity'''
        if self.Bc2vsT_fit is None:
            raise ValueError('to read out data from the Bc2 vs. T relation, execute "calc_diffusivity"')
        if not err:
            return (self.Bc2vsT_fit.Bc2vsT['T'], self.Bc2vsT_fit.Bc2vsT['Bc2'])
        elif err:
            return (self.Bc2vsT_fit.Bc2vsT['T'], self.Bc2vsT_fit.Bc2vsT['Bc2'],
                    self.Bc2vsT_fit.Bc2vsT['T_low_err'], self.Bc2vsT_fit.Bc2vsT['T_upp_err'],
                    self.Bc2vsT_fit.Bc2vsT['Bc2_err'])
        else: raise TypeError('"err" function parameter must be boolean type')

    def calc_diffusivity(self, fit_low_lim=None, fit_upp_lim=None, limitType='T',TcDetermination='Bc2vsTfit',reduced_Bfield='all'):
        '''Input: fit limits (B or T type) of the Bc2(T)
        Output: Diffusivity and related values are set in the properties
        Description: First all B fields (keys) are selected. Then the Bc2vsTfit class is initialized and the diffusivity and related
        values calculated and the respective properties set'''
        B = np.array(list(self.__RT_sweeps_per_B.keys()))
        if reduced_Bfield != 'all':     # only reduce Bfield if fucition is called with other value than 'all'
            B=B[bisect_left(B,min(reduced_Bfield)):bisect_left(B,max(reduced_Bfield))]
        self.Bc2vsT_fit = self.Bc2vsTfit((*self.get_Tc(B=reduced_Bfield, err=True), B), fit_low_lim, fit_upp_lim,limitType)  # initiaizing class
        self.Bc2vsT_fit.calc_Bc2_T_fit(limitType,self.get_R_NC(),self.R_vs_T(B=[0]))  # calculating diffusivity and values
        self._diffusivity, _, _, self._diffusivity_err, _, _ = self.Bc2vsT_fit.get_fit_properties()  # setting properties
        self._Tc_0T = self.Bc2vsT_fit.Tc  # handig over Tc (calculated in Bc2vsTfit)
        return self.Bc2vsT_fit.get_fit_properties()

    def get_Bc2vsT_fit(self, T=None):
        return self.Bc2vsT_fit.linear_fit(T)

    def get_Tc_physical_fit(self,InitialValues=None,return_C_2D=False):
        '''Input: Optional: first guess for Tc and C_2D as tupel. Typical values are (7,1.5)
        Output: Returns Tc at R_max/2 for B=0T according to the physical fit function of Bartolf (2016), p. 188/202
        Description: Fits the RvsT data at B=0 with the physical fit function of Bartolf (2016), p. 188/202. A first guess for Tc
        is made using T at R_max/2. This ensures, that the fit is going to converge.'''
        T, R = self.R_vs_T(B=0)
        Tc_0_Ohm_fit = 0 #Initialize variable #Tc defined as R=0 Ohm
        C_2D = 0 #Initialize variable
        R_max = max(R)
        idx = bisect_left(R, np.max(R)/2)
        T_reduced, R_reduced = (T[idx:], R[idx:])
        if InitialValues==None:
            InitialValues=(T[idx],1.5)    #takes idx element of T, which is at the position of the first R value over R_max/2 (first guess for Tc)
        # define physical fit function according to Bartolf (2016), p. 188/202
        # physical_fit = lambda T,Tc_0_Ohm_fit,C_2D: R_max/(1+R_max*C_2D*1/16*elementary_charge**2/constants.hbar*(T-Tc_0_Ohm_fit)**(-1)*Tc_0_Ohm_fit)  #should i directly write T_reduced instead of T? Should make no difference

        Tc_0_Ohm_fit,C_2D=scipy.optimize.curve_fit(lambda T,Tc_0_Ohm_fit,C_2D: R_max/(1+R_max*C_2D*1/16*elementary_charge**2/constants.hbar*(T-Tc_0_Ohm_fit)**(-1)*Tc_0_Ohm_fit),  T_reduced,  R_reduced, p0=[*InitialValues])[0]
        Tc_R_max_half=(elementary_charge**2/constants.hbar/16*C_2D*R_max+1)*Tc_0_Ohm_fit
        if return_C_2D==True:
            return (Tc_0_Ohm_fit,C_2D) #Returns 0 Ohm Tc for plotting the fitting curve
        else:
            return Tc_R_max_half


    def get_Tc(self, B=None, err=False):
        '''Input: B fields, flag if error should be returned
        Output: Transition temperature Tc of RT-sweep (with lower und upper error)
        Description: Sets B array and checks if corresponding RT sweeps have been fitted. Depending on type of B, returns tuple or tuple of arrays
        with Tc values. Tc comes from the extrapolation of the Bc2 vs T fit.'''
        B = Tools.selector(B, self.default_B_array, np.array(list(self.__RT_sweeps_per_B.keys())))
        B = self.__round_to_decimal(np.array(B), base=self._B_bin_size)
        self.__checkif_B_in_param(B)
        R_NC = self.max_measured_resistance
        if isinstance(B, (int,float)):
            if err is False:
                return self._RTfit.Tc(fit_param=self._parameters_RTfit[B], R_NC=R_NC)[0]
            else:
                return self._RTfit.Tc(fit_param=self._parameters_RTfit[B], R_NC=R_NC)
        else:
            Tc, Tc_err_low, Tc_err_up = (np.array([]), np.array([]), np.array([]))
            for key in B:
                Tc= np.append(Tc, self._RTfit.Tc(fit_param=self._parameters_RTfit[key], R_NC=R_NC)[0])
                Tc_err_low = np.append(Tc_err_low, self._RTfit.Tc(fit_param=self._parameters_RTfit[key], R_NC=R_NC)[1])
                Tc_err_up = np.append(Tc_err_up, self._RTfit.Tc(fit_param=self._parameters_RTfit[key], R_NC=R_NC)[2])
            if err is False:
                return Tc
            else: return (Tc, Tc_err_low, Tc_err_up)

    def get_transition_width(self,B=None,calculationStyle='Data'):
        '''Returns the width of the superconducting transition. It gets therefore the temperatures for 10% and 90% of R_normal_conducting
        and the difference between those is the width of the superconducting transition.'''
        if calculationStyle=='Data':
            if B==None:
                t,r = self.R_vs_T(B=0)
                idx_10 = bisect_left(r, 0.1*np.max(r))  #10% R_max
                idx_90 =  bisect_left(r, 0.9*np.max(r))  #90% R_max
                if (idx_10 == 0) or (idx_10 >= np.size(t)-1) or (idx_90 == 0) or (idx_90 >= np.size(t)-1):
                        raise IndexError('Not enough data points for extracting 10% or 90% of R_max. Possibly too restricted fitting range or empty array.')
                #Calculating the exact R_max/2 point that is between the datapoints around the real R_max/2 value
                #to get more accurate outcomes
                slope_10 = (r[idx_10]-r[idx_10-1])/(t[idx_10]-t[idx_10-1])
                y_intercept_10 = r[idx_10]-slope_10*t[idx_10]
                T_10percent_R_max=(0.1*np.max(r)-y_intercept_10)/slope_10

                slope_90 = (r[idx_90]-r[idx_90-1])/(t[idx_90]-t[idx_90-1])
                y_intercept_90 = r[idx_90]-slope_90*t[idx_90]
                T_90percent_R_max=(0.9*np.max(r)-y_intercept_90)/slope_90
                width = T_90percent_R_max-T_10percent_R_max
                return width

            else:  #input is B field range
                T,R = self.R_vs_T(B)
                width = []
                B_field=[]
                for key, t, r in zip(T.keys(), T.values(), R.values()):
                    idx_10= bisect_left(r, 0.1*np.max(r))  #10% R_max
                    idx_90 =  bisect_left(r, 0.9*np.max(r))  #90% R_max
                    if (idx_10 == 0) or (idx_10 >= np.size(t)-1) or (idx_90 == 0) or (idx_90 >= np.size(t)-1):
                            raise IndexError('Not enough data points for extracting 10% or 90% of R_max. Possibly too restricted fitting range or empty array.')
                    #Calculating the exact R_max/2 point that is between the datapoints around the real R_max/2 value
                    #to get more accurate outcomes
                    slope_10 = (r[idx_10]-r[idx_10-1])/(t[idx_10]-t[idx_10-1])
                    y_intercept_10 = r[idx_10]-slope_10*t[idx_10]
                    T_10percent_R_max=(0.1*np.max(r)-y_intercept_10)/slope_10

                    slope_90 = (r[idx_90]-r[idx_90-1])/(t[idx_90]-t[idx_90-1])
                    y_intercept_90 = r[idx_90]-slope_90*t[idx_90]
                    T_90percent_R_max=(0.9*np.max(r)-y_intercept_90)/slope_90

                    width.append(T_90percent_R_max-T_10percent_R_max)
                    # B_field.append(key)
                return width

        if calculationStyle=='Fit':
            B_values=np.arange(0,1.1,self._B_bin_size) #need way to take the B_sweep borders of the measurement
            trans_widths=self.get_transition_width(B=B_values,calculationStyle='Data')
            lin_fit = np.polyfit(B_values,trans_widths ,1)
            return lin_fit     #should return slope and width

    def get_transport_properties(self,film_thickness,fit_range=[0.4,0.9]):
        '''Input: film film thickness in nm! and optional the fit range in Tesla.
         Output: dictionary with all possible parameters that can be calculated after transport measurements'''
        d_values=self.calc_diffusivity(fit_range[0],fit_range[1],limitType='B') #calculate diffusivity like in jupyter notebook
        D=d_values[0]
        D_err=d_values[3]
        dBc2dT=d_values[1]
        Tc=self.get_Tc_physical_fit()
        R_sheet_20K=self.get_R_NC()
        film_thickness=film_thickness*1e-9 #convert nm into m

        energy_gap=1.76*Boltzmann*Tc  #1.76 comes from BCS ratio, it is not fixed but dependent on R_sheet (in principal)
        Bc2=0.69*Tc*dBc2dT      #0.69*Tc*dBc2dT
        dos=1e4/(elementary_charge**2*D*film_thickness*R_sheet_20K) #in 1/Jm³   1e4 because of D in cm²/s
        resistivity_20K=film_thickness*R_sheet_20K #in Ohm*m units  #Ohm*m=1e6µOhm*1e2cm=1e8µOhm*cm
        density_free_energy_difference=0.5*dos*energy_gap**2 #in 1/m³
        transition_width_0T=self.get_transition_width()  #since currently it is only possible to get transWidths from
                                                                   #this function, give [0,0]and take the first value
        depairing_current_density_0K = 0.74*(energy_gap**1.5)/(elementary_charge*R_sheet_20K*np.sqrt(constants.hbar*D*1e-4)*film_thickness)
        ###D is in cm²/s --> multiply D with 1e-4 for SI Units, dep. curr. density is in A/m units
        coherence_length=np.sqrt(constants.hbar*D/energy_gap)         #Unit is cm

        # return (depairing_current_density_0K,dos,energy_gap,coherence_length,Bc2,density_free_energy_difference,
        # resistivity_20K,transition_width_0T,film_thickness)#also return Tc, D,etc?
        return {'Diffusivity':D,'DiffusivityError':D_err,'Tc':Tc,'SheetResistance':R_sheet_20K,'DepCurrDens':depairing_current_density_0K,'DensityOfStates':dos,'EnergyGap':energy_gap,
        'CoherenceLength':coherence_length,'Bc2':Bc2,'DensFreeEnergyDiff':density_free_energy_difference,'Resistivity':resistivity_20K,
        'TransitionWidth':transition_width_0T,'Thickness':film_thickness}

    def get_hall_properties(self,film_thickness):
        dRdB=self.get_R_hall_fit()
        hall_coefficient=abs(dRdB)*film_thickness
        electron_density=1/(film_thickness*abs(dRdB)*elementary_charge) #unit 1/m³
        return {'ElectronDensity':electron_density,'Hall Coefficient':hall_coefficient,'Slope':dRdB}

    def get_R_hall_fit(self):
        all_B_values=[]
        all_R_values=[]
        for i in range(0,len(self._B_sweeps)):
            B, R = self.R_vs_B(i)
            all_B_values=np.concatenate((all_B_values,B))   #combining all sweeps in 2 arrays (B,R)
            all_R_values=np.concatenate((all_R_values,R))
        dBdR,R_0T=np.polyfit(all_B_values, all_R_values , 1)  #fitting linear dooh
        return dBdR

    def fit_function_parameters(self, B=None):
        '''Input: B fields as array or scalar
        Output: fitting parameters for chosen B fields
        Description: Sets chosen B fields and returns dictionary of fit parameters with B fields as keys'''
        B = Tools.selector(B, self.default_B_array, np.array(list(self.__RT_sweeps_per_B.keys())))
        B = self.__round_to_decimal(np.array(B), base=self._B_bin_size)  # This and above line needed to assure B values are valid and rounded to correct decimal
        if isinstance(B, (int, float)):
            self._RTfit.read_RT_data(self.__RT_sweeps_per_B[B])
            self._parameters_RTfit[B] = self._RTfit.fit_data(fit_low_lim=self.__RT_fit_low_lim, fit_upp_lim=self.__RT_fit_upp_lim)
            return self._parameters_RTfit[B]
        # Difference between scalar and array to be able to return dict or dict of dicts
        elif isinstance(B, (np.ndarray, list)):
            return_fit_param_dict = {}
            for k in B:
                self._RTfit.read_RT_data(self.__RT_sweeps_per_B[k])
                self._parameters_RTfit[k] = self._RTfit.fit_data(fit_low_lim=self.__RT_fit_low_lim, fit_upp_lim=self.__RT_fit_upp_lim)
                return_fit_param_dict[k] = self._parameters_RTfit[k].copy()
            return return_fit_param_dict
        else: raise TypeError('input parameters must have correct type. check input parameters')

    def calc_RT_fits(self,BFieldRange='all'):               #to prevent unreadable data caused by bad RvsT curves at specific B field
        self.fit_function_parameters(BFieldRange)

    # def set_RT_fit_limits(self, fit_low_lim, fit_upp_lim):  # sets upper and lower limit for the fits of RT sweeps
    #     self.__RT_fit_low_lim = fit_low_lim
    #     self.__RT_fit_upp_lim = fit_upp_lim

    def get_RT_fit_limits(self): # gets upper and lower limit for the fits of RT sweeps
        return (self.__RT_fit_low_lim, self.__RT_fit_upp_lim)

    def fit_function(self, B=None, T=None):
        '''Input: B field as array or scalar, Temperature T as array, dict or scalar
        Output: tuple of arrays, scalars or dicts of the fitted RT values
        Description: Sets B to valid values and checks if fit parameters exist. Varying tuple entries depending on input of B and T.
        For T=None, the T array is returned as well. Possible to hand over personalized T arrays or the T arrays of the measurement corresponding to the B fields entered'''
        B = Tools.selector(B, self.default_B_array, np.array(list(self.__RT_sweeps_per_B.keys())))
        B = self.__round_to_decimal(np.array(B), base=self._B_bin_size)
        self.__checkif_B_in_param(B)
        if isinstance(B, (int, float)):
            return self._RTfit.return_RTfit(*self.set_temperature_array(T), self._parameters_RTfit[B])
        elif isinstance(T, (dict)) and isinstance(B, (np.ndarray, list)):
            if not set(list(T.keys())).issubset(B):  # used to check if B fields of T array (keys) are included in the given B values
                raise TypeError("B values are not matching. Please revise B values of temperature dictionary")
            for t, k in zip(T.values(), B):
                T_data, return_T = self.set_temperature_array(t)
                self._fitted_RTvalues[k] = self._RTfit.return_RTfit(T_data, return_T, self._parameters_RTfit[k])
        else:
            for k in B:
                T_data, return_T = self.set_temperature_array(T)
                self._fitted_RTvalues[k] = self._RTfit.return_RTfit(T_data, return_T, self._parameters_RTfit[k])
            self._fitted_RTvalues = self.__unpack_tuple_dictionary(self._fitted_RTvalues)
        if return_T:  # self._fitted_RTvalues is a dictionary with tuples as values. To keep systematic, a tuple of dicts must be returned, as happend here
            T_fit, R_fit = ({}, {})
            for key, value in self._fitted_RTvalues.items():
                T_fit[key], R_fit[key] = (value['T'], value['R'])
            return (T_fit, R_fit)
        else: return self._fitted_RTvalues

    def set_temperature_array(self,T):
        '''Input: T array
        Output: Adapted T array
        Description: Sets T array correctly according to specifications: if None: maximal T range of measurement, else return T. Also, this method sets
        if T array should be returned as well in fit_function'''
        request_eval_array = False
        if T is None:
            T_min = np.inf; T_max = 0
            for value in self.__RT_sweeps_per_B.values():  # looking for max (min) values
                if np.min(value['T']) < T_min:
                    T_min = np.min(value['T'])
                if np.max(value['T']) > T_max:
                    T_max = np.max(value['T'])
            T_array = np.linspace(T_min, T_max, num=self._RTfit.fit_function_number_of_datapoints)
            request_eval_array = True
        elif isinstance(T, (list, np.ndarray)):
            T_array = T
        elif isinstance(T, (int, float)):
            return (T, request_eval_array)
        else: raise TypeError('input parameters must have correct type. check input parameters')
        return (T_array, request_eval_array)

    def __checkif_B_in_param(self, B):
        '''Input: B array
        Output: -
        Description: Checks whether B fields in B array have already been fitted, otherwise calls fit_function_parameters'''
        not_in_param_scalar = isinstance(B, (int,float)) and B not in self._parameters_RTfit.keys()
        not_in_param_array = isinstance(B, (list, np.ndarray)) and not set(B).issubset(set(list(self._parameters_RTfit.keys())))
        if not_in_param_scalar or not_in_param_array:
            self.fit_function_parameters(B)
        elif not isinstance(B, (int, float, list,np.ndarray)):
            raise TypeError('input parameters must have correct type. check input parameters')

    def __unpack_tuple_dictionary(self, dict_input):  # returns dictionary with keys for T and R taken from the dictionary of tuples
        return_dict = {}
        for key, value in dict_input.items():
            return_dict[key] = {'T': value[0], 'R': value[1]}
        return return_dict


    class Bc2vsTfit():
        """This class object contains the fit of the Bc2-T relation, performing all necessary steps to calculate up to the diffusivity.
        It takes as input the Bc2 and T arrays determined from the RT fitting. These values are fitted with linear regression and the
        diffusivity and related values are calculated and stored within the class object. It is a nested class from DiffusivityMeasurement."""

        def __init__(self, data, fit_low_lim, fit_upp_lim,limitType):
            self.__D = 0
            self.__dBc2dT = 0
            self.__B_0 = 0
            self.__err_D = 0
            self.__err_dBc2dT = 0
            self.__r_squared = 0
            self.Bc2vsT = {}  # here the (T,Bc2) data points determined from the measurement and fitting are stored
            self.Tc = 0

            self.B_field_meas_error_pc = 0.02 # variation of 2% in voltage monitor results in variation of 1% in Bfield
            self.B_field_meas_error_round = 0.001 # tbreviewed, in Tesla, measurement error + rounding error
            self.linear_fit_number_of_datapoints = 100  # amount of data points to be placed between lower and upper T values for the linear fit

            self.set_properties(data)  # read data into attributes

            self.low_lim = Tools.selector(fit_low_lim, np.sort(self.Bc2vsT['T'])[1])  # determine lower fit limit, if None the minimum of T is taken
            self.upp_lim = Tools.selector(fit_upp_lim, np.sort(self.Bc2vsT['T'])[-2]) # determine upper fit limit, if None the maximum of T is taken


    # @property
    # def Tc(self):
    #     return self._Tc




        def set_properties(self, data): # set the attributes of the class feeding them with data provided from the outer class
            self.Bc2vsT['T'], self.Bc2vsT['T_low_err'], self.Bc2vsT['T_upp_err'], self.Bc2vsT['Bc2'] = data
            self.Bc2vsT['Bc2_err'] = self.Bc2vsT['Bc2']*self.B_field_meas_error_pc + self.B_field_meas_error_round

        def linear_fit(self, T=None):
            '''Input: Temperature values to be evaluated for Bc2
            Output: Bc2-array or (T,Bc2) tuple
            Description: takes the T array and, using the slope and the intercept calculated, evalates T values.'''
            T_min_def = np.min(self.Bc2vsT['T'])
            T_max_def = np.max(self.Bc2vsT['T'])
            if T is None:
                T = Tools.selector(T, np.linspace(T_min_def, T_max_def, num=self.linear_fit_number_of_datapoints))
                return (T, self.__dBc2dT*T + self.__B_0)
            elif isinstance(T, (list, np.ndarray)):
                return self.__dBc2dT*T + self.__B_0
            else:
                raise TypeError('wrong type of input parameter T. Please check input parameters')

        def calc_Bc2_T_fit(self,limitType,R_NC,RT_0T):
            '''Input: uses attributes from the class object
            Output: sets attributes of the class object
            Description: Defines T-Bc2 values to be fitted through lower and upper fit limit. Calculates the fit with linear regression.
            Calculates diffusivity, diffusivity error and Tc(0T).'''

            #T_fit_array, Bc2_fit_array = Tools.select_values(self.Bc2vsT['T'], self.Bc2vsT['Bc2'], self.low_lim, self.upp_lim)

            if limitType=='T':
                T_fit_array, Bc2_fit_array = Tools.select_values(self.Bc2vsT['T'], self.Bc2vsT['Bc2'], self.low_lim, self.upp_lim)
            elif limitType=='B':
                Bc2_fit_array, T_fit_array = Tools.select_values(self.Bc2vsT['Bc2'], self.Bc2vsT['T'], self.low_lim, self.upp_lim)
            else:
                raise Exception('Wrong limit type: Please use T for Temperature fitting limits and B for B field fitting limits.')


            if T_fit_array.size == 0:
                raise ValueError('chosen fit limits result in empty array. please change fit limits')
            self.__dBc2dT, self.__B_0, r_value, _, self.__err_dBc2dT = \
                linregress(T_fit_array, Bc2_fit_array)
            self.__r_squared = r_value**2
            self.__D = -4*Boltzmann/(pi*elementary_charge*self.__dBc2dT)*1e4
            self.__err_D = abs(4*Boltzmann/(pi*elementary_charge*(self.__dBc2dT**2))*self.__err_dBc2dT)*1e4 +  0.0373 * self.__D # from gaussian error propagation:
            #total Error = error_BC2vsT_fit + statisitical error from std dev (gained from reproducibility measurements) (percentage error)

            self._Tc = -self.__B_0/self.__dBc2dT #intercept of Bc2vsT curve with x max_measured_resistance

        def get_fit_properties(self):  # returns the fit properties/physical properties since they are private
            return (self.__D, self.__dBc2dT, self.__B_0, self.__err_D, self.__err_dBc2dT, self.__r_squared)


class RTfit():
    """This class object describes an RT fit, including all necessary parameters, attributes and methods to determine an RT fit.
    Data is read into the object as T and R arrays. Data is fitted with gaussian cdf/richards function. Fit limits for the RT fits can be defined.
    Fit can be returned as T and R arrays. Transition temperature Tc can be returned (optionally with error)"""

    def __init__(self, fit_function_type = "R_max/2"):
        self._fit_function_type = fit_function_type
        # self.fit_function = self.richards # BUG? This line should be removed and the following 6 lines be there instead, right?
        if self._fit_function_type == "richards":
            self.fit_function = self.richards
        elif self._fit_function_type == "gauss_cdf":
            self.fit_function = self.gauss_cdf
        elif self._fit_function_type == "R_max/2":
            self.fit_function = self.linear_function
        # elif self.fit_function == "physical fit":  # Tc from ohysical fit has its own function: get_Tc_physical_fit()
        #     self.fit_function = self.physical_fit
        else:
            raise ValueError('only "R_max/2", "richards", "physical fit" and "gauss_cdf" as possible fitting functions')
        self.fit_function_number_of_datapoints = 1000  # amount of data points to be placed between lower and upper T values for richards/gauss_cdf fit
        self._T = 0
        self._R = 0
        self._fit_param = {'output':{}, 'start_values':{}}  # output are always the resulting fit parameters after fitting, start_values represent
                                                           # the fit parameters values used in the beginning by the fit function
        self.curve_fit_options = {}  # describes further fit options necessary for the fitting
        self.fit_covariance_matrix = {}  # matrix containing implicitly fit error
        self.__set_fit_parameters = {}  # fit parameters set by the user, default is empty

        self.T_meas_error_pc = 0.0125 # in percent, estimated interpolation error
        self.T_meas_error_std_dev = 0.02 # standard deviation of measurements at 4Kelvin
        self.R_meas_error_pc = 0.017 # relative resistance error from resistance measurement

    @property
    def fit_function_type(self):
        return self._fit_function_type

    @property
    def T(self):
        return self._T

    @property
    def R(self):
        return self._R

    @property
    def fit_param(self):
        return self._fit_param

    def read_RT_data(self, data):
        '''Input: data as dict, numpy array, tuple
        Output: sets attributes self._T and self._R
        Description: based on the type of data, the method is able to read out the data in several formats including:
        dict: {'T':x, 'R':y} ; tuple: (T,R)
        numpy.array: [[x,y], [x,y]] or [[xxx], [yyy]]'''
        try:
            if type(data) is dict:
                self._T, self._R = (np.asarray(data['T']), np.asarray(data['R']))
            elif type(data) is np.ndarray:
                shape = np.shape(data)
                if shape[0] is 2:
                    self._T, self._R = (np.asarray(data[0,:]), np.asarray(data[1,:]))
                elif shape[1] is 2:
                    self._T, self._R = (np.asarray(data[:,0]), np.asarray(data[:,1]))
                else: raise ValueError('array has the shape ' + str(shape))
            elif type(data) is tuple:
                self._T, self._R = data
        except: raise Exception('data has not the correct format or is empty. please check passed data')

    def fit_data(self, fit_low_lim=None, fit_upp_lim=None, R_NC=None):
        '''Input: fit limits
        Output: fit parameters for one RT sweep as dictionary
        Description: reduces the array according to fit limits. Depending on fit function, sets correct fitting parameters and calculates fit with curve_fit.'''
        T, R = Tools.select_values(self._T, self._R, fit_low_lim, fit_upp_lim)
        #print(self._fit_function_type)
        if T.size == 0:
            raise ValueError('chosen fit limits result in empty array. please change fit limits')
        if self._fit_function_type == "richards":
            self.__define_fitting_parameters_richards(R_NC, **self.__set_fit_parameters)
            self.fit_function = self.richards
        elif self._fit_function_type == "gauss_cdf":
            self.__define_fitting_parameters_gauss_cdf(R_NC, **self.__set_fit_parameters)
            self.fit_function = self.gauss_cdf
        elif self._fit_function_type == "R_max/2":
            # idx = self._T[bisect_left(self._R, Tools.selector(R_NC, np.max(self._R))/2)]  # index of temperature just below (or equal) where resistance reaches half its
            idx = bisect_left(self._R, Tools.selector(R_NC, np.max(self._R))/2)  # index of temperature just below (or equal) where resistance reaches half its normal conducting value
            if (idx == 0) or (idx >= np.size(T)-1):
                raise IndexError('Not enough data points for extracting R_max/2. Possibly too restricted fitting range or empty array.')
            #print(idx)
            T, R = (self._T[idx-1:idx+1], self._R[idx-1:idx+1]) # reduce arrays to fit to just 2 data points for linear interpolation
            self.__define_fitting_parameters_R_max_half(**self.__set_fit_parameters)
            self.fit_function = self.linear_function

        else: raise ValueError('only "richards", "gauss_cdf" and "R_max/2" as possible fitting functions')

        #############################
        # BUG? In the next line, should it not be T and R instead of self._T and self._R because otherwise the range is not restricted to the selected one, right?
        #############################
        popt, self.fit_covariance_matrix = curve_fit(self.fit_function, self._T, self._R, list(self._fit_param['start_values'].values()), **self.curve_fit_options)
        #print(popt)
        popt, self.fit_covariance_matrix = curve_fit(self.fit_function, T, R, list(self._fit_param['start_values'].values()), **self.curve_fit_options)
        #print('Curve fit over selected T')
        #print(popt)
        self._fit_param['output'] = {key: value for key, value in zip(self._fit_param['start_values'].keys(), popt)}
        return self._fit_param['output']

    def __define_fitting_parameters_richards(self, R_NC=None, **kwargs):
        '''Input: normal conducting resistance, optionally fitting parameters
        Output: sets starting values of fit parameters (attribute)
        Description: defines the start values if the richards function is selected. if several fits are performed, the resulting fit parameters_RTfit
        from the fit before are used as starting values'''
        R_NC = Tools.selector(R_NC, np.max(self._R))
        a,c,q = (0,1,1)
        if self._fit_param['output'] == {} and kwargs == {}:  # use calculated starting values if no fit has been already performed
            k = R_NC  # upper asymptote
            nu = 1  # affects near which asymptote maximum growth occurs (nu is always > 0)
            m = a + (k-a)/np.float_power((c+1),(1/nu))  # shift on x-axis
            t_2 = self._T[bisect_left(self._R, R_NC/2)]  # temperature where resistance reaches half its normal conducting value
            b = 1/(m-t_2) * ( np.log( np.float_power((2*(k-a)/k),nu)-c )+np.log(q) ) # growth rate
            self._fit_param['start_values'] = {'b': b, 'm': m, 'nu': nu, 'k': k}
        elif kwargs != {}:  # use parameters set by user
            self._fit_param['output'] = {}
            self.__define_fitting_parameters_richards()  # call function again to ensure missing all fitting parameters are existing
            for key, value in kwargs.items():
                if key in ['b', 'm', 'nu', 'k']:
                    self._fit_param['start_values'][key] = value  # overwrite values of keys handed over to the function in kwargs
            self.__set_fit_parameters = {}
        else:
            self._fit_param['start_values'] = self._fit_param['output']
        self.curve_fit_options = {'maxfev': 2500, 'bounds': ([-np.inf, -np.inf, -np.inf, 0.8*R_NC], [np.inf, np.inf, np.inf, 1.2*R_NC])}

    def __define_fitting_parameters_gauss_cdf(self, R_NC=None, **kwargs):
        '''Input: normal conducting resistance, optionally fitting parameters
        Output: sets starting values of fit parameters (attribute)
        Description: defines the start values if the gaussian cdf function is selected. calculates the starting values with every new self._R, self._T handed over to the class'''
        R_NC = Tools.selector(R_NC, np.max(self._R))
        if bisect_left(self._R, R_NC/2) < len(self._R):  # perform check whether the searched index for the temperature of the halfpoint value is realistic
            mean = self._T[bisect_left(self._R, R_NC/2)]  # is below the maximal length, otherwise the array is sorted the other way round and bisect left gives unrealistic indices
            sigma = self._T[bisect_left(self._R, 0.9*R_NC)]-self._T[bisect_left(self._R, 0.1*R_NC)]
        else:
            R_rev = self._R[::-1]  # turn around the array and search again for the temperature value at the halfpoint resistance, as check above was negative
            mean = self._T[bisect_left(R_rev, R_NC/2)]
            sigma = self._T[bisect_left(R_rev, 0.9*R_NC)]-self._T[bisect_left(R_rev, 0.1*R_NC)]
        if sigma < 0.01:  # ensure sigma has a good starting value
            sigma = 0.1
        self._fit_param['start_values'] = {'scaling': R_NC, 'mean': mean, 'sigma': sigma}
        if kwargs != {}:  # if fit starting values are handed over by the user, use them!
            self.__define_fitting_parameters_gauss_cdf()  # ensure there are the necessary values
            for key, value in kwargs.items():  # overwrite starting values handed over by the user
                if key in ['scaling', 'mean', 'sigma']:
                    self._fit_param['start_values'][key] = value
            self.__set_fit_parameters = {}
        self.curve_fit_options = {'maxfev': 1600, 'bounds': (-inf, inf)}

    def __define_fitting_parameters_R_max_half(self, **kwargs):
        '''Input: Optionally starting values for linear fit
        Output: sets starting values of fit parameters (attribute)
        Description: defines the start values if we want a simple linear interpolation between closest data
        points to R_max/2. Calculates the starting values with every new self._R, self._T handed over to the class'''
        self._fit_param['start_values'] = {'slope': 1.0, 'yintercept': -100}
        if kwargs != {}:  # if fit starting values are handed over by the user, use them!
            self.__define_fitting_parameters_R_max_half()  # ensure there are the necessary values
            for key, value in kwargs.items():  # overwrite starting values handed over by the user
                if key in ['slope', 'yintercept']:
                    self._fit_param['start_values'][key] = value
            self.__set_fit_parameters = {}
        self.curve_fit_options = {'maxfev': 1600, 'bounds': (-inf, inf)}

    def set_fit_parameters(self, **kwargs):  # sets the fit parameters as method for comfortness
        self.__set_fit_parameters = kwargs

    def richards(self, t,b,m,nu,k, a=0, c=1, q=1):
        return a + (k-a)/np.float_power((c+q*np.exp(-b*(t-m))),1/nu)

    def gauss_cdf(self, x, scaling, mean, sigma):
        return scaling*norm.cdf(x, mean, sigma)

    def linear_function(self, x, slope, yintercept):
        return slope*x+yintercept

    def return_RTfit(self, eval_array = None, return_eval_array = False, fit_param=None):
        '''Input: T array to be evaluated, flag if eval_array should be returned, which fit parameters to use
        Output: R array or tuple (T,R) of fitted values with the given parameters
        Description: takes or sets an evaluation array and evaluates them in the given function with the passed fitting parameters'''
        fit_param = Tools.selector(fit_param, self._fit_param['output'])
        if eval_array is None:  # set evaluation array limits as the limits of the measured self._T array
            x_low_lim = np.min(self._T)
            x_upp_lim = np.max(self._T)
            eval_array = np.linspace(x_low_lim, x_upp_lim, self.fit_function_number_of_datapoints)
        if return_eval_array:  # eval array should be returned
            return (eval_array, self.fit_function(eval_array, **fit_param))
        else:
            return self.fit_function(eval_array, **fit_param)

################ maybe insert here R_NC as parameter to give over to TC function? and then call get_TC_RMax2 with that
    def Tc(self, fit_param = None, R_NC = None):
        '''Input: fit parameters to use for determining the according Tc value of the RT-sweep
        Output: tuple (Tc, Tc_err_low, Tc_up_err): it is the transition temperature of an RT sweep
        Description: depending on the fit function, calls corresponding private method and returns Tc with the error'''
        fit_param = Tools.selector(fit_param, self._fit_param['output'])
        if self._fit_function_type == 'richards':
            Tc = self.__get_Tc_richards(fit_param)
            return (Tc, *self.__Tc_error(Tc))
        elif self._fit_function_type == 'gauss_cdf':
            Tc = self.__get_Tc_gauss_cdf(fit_param)
            return (Tc, *self.__Tc_error(Tc))
        elif self._fit_function_type == 'R_max/2':
            if R_NC == None: raise ValueError('No argument R_NC passed to RTfit.Tc although necessary for calculating Tc with R_max/2 method.')
            Tc = self.__get_Tc_R_max_half(fit_param, R_NC)
            return (Tc, *self.__Tc_error(Tc))
        else: raise ValueError('only "richards", "gauss_cdf", and "R_max/2" as possible fitting functions')

    def __get_Tc_richards(self, param):  # the halfpoint value of the richards function is analytically calculated and is then returned
        if {'b', 'm', 'nu', 'k'}.issubset(list(param.keys())):
            a,c,q = (1,1,1)
            b, m, nu, k = param.values()
            return m - 1/b*( np.log(np.float_power(2*(k-a)/k,nu)-c) + np.log(q) )  # analytical calculation of the halfpoint
        else:
            raise ValueError('no richards parameters found')

    def __get_Tc_gauss_cdf(self, param):  # the Tc of the gaussian fit is described by the expected mean mu, since this always describes the halfpoint of a gaussian cdf
        if 'mean' in param.keys():
            return param['mean']
        else: raise ValueError('no gaussian parameters found')

    def __get_Tc_R_max_half(self, param, R_NC):  # the Tc is simply given by linear interpolation
        if {'slope', 'yintercept'}.issubset(list(param.keys())):
            return (R_NC/2 - param['yintercept'])/param['slope']
        else: raise ValueError('no R_max/2 parameters found')



    def __Tc_error(self, Tc):
        '''Input: calculated Tc
        Output: Tuple (Tc_err_low, Tc_err_up)
        Description: Tc error is defined as the interval between two measured data points in which the calculated value lies, with lower and upper error respectively.
        Additionally, measurement errors are considered'''
        # print('length T:',len(self._T))  #DEBUG
        # print('bisect_left indices: ',bisect_left(self._T, Tc) - 1,' and ',bisect_left(self._T, Tc))
        T_data_from_below = self._T[bisect_left(self._T, Tc) - 1]
        T_data_from_above = self._T[bisect_left(self._T, Tc)]
        T_err_low = abs(Tc - T_data_from_below - self.T_meas_error_pc * T_data_from_below - self.T_meas_error_std_dev) # consideration of measurement errors
        T_err_up = abs(T_data_from_above + self.T_meas_error_pc * T_data_from_above + self.T_meas_error_std_dev - Tc)
        return (T_err_low, T_err_up)


class Tools():

    @staticmethod
    def selector(val, *args): # function selecting out of the passed args according to the control structure. outsourced since it was used a lot
        if val is None:       # to hand over properties, as they cannot be set as default values in the parameter definition
            return args[0]
        elif val == 'all':
            return args[1]
        else: return val

    @staticmethod
    def select_values(X, Y, fit_low_lim, fit_upp_lim):  # function reducing an x-Y relation in X dimension according to fit limits. Needed to reduce the fit area of RT sweeps and Bc2vsT relation
        XY_data = np.array([X, Y]).transpose()  # build a matrix and transpose it to have X,Y pairs
        XY_data = XY_data[XY_data[:,0].argsort()]  # sort the matrix in X dimension
        XY_data = XY_data[XY_data[:,0] >= fit_low_lim, :]  # check for lower limit
        XY_data = XY_data[XY_data[:,0] <= fit_upp_lim, :]  # check for upper limit
        T_array, Bc2_array = (XY_data[:,0], XY_data[:,1])  # create tuple of (X,Y)
        return (T_array, Bc2_array)
