import os, re
import json
import datetime
import pandas as pd
import numpy as np

from pytz import timezone

import EDT_ID_lime as id_lime
import CP_Global_Utils as G_Utils
import CP_Global_Config as G_config
import ESC_Data_Management as Data_mgt
import ESC_Kiln_Management as Kiln_mgt
import ESC_ES_recommendations as ES_recommendations
import ESC_heat_input_saturations as hi_saturations
import ESC_Utils as Utils

import pickle, time

import warnings

warnings.filterwarnings("ignore")


class Controller(G_config.ConfigFactory):
    """
    A class to represent a controller.

    Attributes
    ----------

    Methods
    -------
    extract_data():
        extract data and all configuration files.
    write_results(self, json_param, results, output_json, results_table):
        description
    run():
        extract data, run the controller and write the results.
    """

    def __init__(self, **kwargs):
        """
        Constructs all the necessary attributes for the expert system controller object.
        """
        super(Controller, self).__init__(**kwargs)
        self.config_model_management()
        self.config_controller()
        self.config_anomaly_detection()
        self._import_distribution_data()
        self._import_ad_data()

        self.id_cols = ['Timestamp', 'Cycle - Number', 'Calculation - Kiln pressure drop ratio - Last cycle',
                        'Production rate', 'Heat input - Setpoint',
                        'Solid fuel transport air - Theoretical air flow from solid fuel transport air blower'
                        'Cooling air - Coefficient',
                        'Solid fuel transport air - Theoretical air flow from solid fuel transport air blower',
                        'Cooling air - Coefficient',
                        'Rolling_average_HI',
                        'Max_of_avg_pyro_reversal', 'Max_of_avg_TT_comb',
                        'Combustion air - Excess air',
                        'Outside_temp'
                        ]

        self.id_cols_since_cz = ["Heat input - Setpoint", "Combustion air - Excess air"]

    def extract_data(self):

        try:
            with open(self.filepath_data_transfos, 'rb') as handle:
                self.orig_dict = pickle.load(handle)
                for k in self.orig_dict.keys():
                    if isinstance(self.orig_dict[k][0], float) & (k != 'Comment') & ('Peak' not in k) & \
                            ('Valley' not in k):
                        self.orig_dict[k] = self.orig_dict[k].astype(np.float)

            with open(self.filepath_models_management, "r") as externalFile:
                self.write_to_opc_management = json.load(externalFile)

            if os.path.isfile(self.filepath_results):
                self.results = pd.read_csv(self.filepath_results, sep=',', header='infer', encoding='utf-8')
                self.model_in_use = self.results["Model_in_use"].iloc[-1]
            else:
                self.results = pd.DataFrame()
                self.model_in_use = 'Expert System'

            return

        except Exception as e:
            log = "error when importing data:" + '\n' + str(e) + "\n"
            print(log)
            title = self.mail_title_prefix + ' ' + self.kiln + ': error when importing data'
            Utils.send_mail(title, log, self.json_param, intern=True)
            output_json = "notprocessed"
            return output_json

    def compute_controller_recommendations(self, print_log=True):

        t0 = time.time()
        enough_qual = True
        data_dict = self.orig_dict.copy()

        data_dict.update(id_lime.get_smart_aggregations(data_dict, self.id_cols, 'avg_bz', sample_ref='co2'))
        data_dict.update(id_lime.get_smart_aggregations(data_dict, self.id_cols_since_cz, 'avg_since_cz', 'co2'))
        if "sticky" in data_dict.keys():
            data_dict.update(id_lime.get_smart_aggregations(data_dict, self.id_cols_since_cz, 'avg_since_cz', 'tc'))

        if 'Cycle - Number' not in self.results.columns:
            self.results.rename(columns={'Cycle': 'Cycle - Number'}, inplace=True)
        last_processed_cycle = self.results['Cycle - Number'].iloc[-1] if len(self.results) > 0 else 0

        log_data_esc = Utils.ExpertSystemLogging(self.model_in_use, self.json_param['HI_units'])
        if 'new_calc' not in self.results.columns:
            cycle_diff_last_eval = 0
        else:
            cycle_diff_last_eval = len(self.results['new_calc']) - np.where(self.results['new_calc'] == 1)[0][-1]
        try:
            ######################################################
            ''' get some process parameters'''
            ######################################################
            # get from results
            last_processed_sample_cycle = \
                pd.Series(self.results["Last_sample_cycle"]).fillna(method='ffill').to_numpy()[-1] \
                    if len(self.results) > 0 else 0
            time_now = datetime.datetime.now(timezone('UTC')).isoformat()

            #####################################################################
            ''' Get quality samples for evaluation '''
            #####################################################################

            data_dict, qual_dict, enough_qual = Data_mgt.get_quality_samples_for_evaluation(data_dict)

            #####################################################################
            ''' Get all Kiln/Process information '''
            #####################################################################

            escm = Kiln_mgt.ExpertSystemChangeManagement(model_in_use=self.model_in_use)
            data_dict = escm.change_manager(data_dict, self.json_param, qual_dict['last_qualities'],
                                            last_processed_sample_cycle, cycle_diff_last_eval)

            #########################
            ''' Check Kiln state '''
            #########################
            # param_out, self.log_dict = \
            #     Kiln_mgt.kiln_state_control(data_dict, self.json_param, self.filepath_ad_table,
            #                                 context_row, self.log_dict)

            # [ad_load_withdrawal, ad_reducing_hi, ad_mail4_concern] = param_out
            # [ad_load_withdrawal, ad_reducing_hi, ad_mail4_concern] = False, False, False
            # log_data_esc.write_kiln_state_control(self.log_dict)

            #####################################################################
            ''' Check whether or not only one kiln is allowed to write in opc'''
            #####################################################################
            model_writing, enable_opc_write = Kiln_mgt.write_to_opc_manager(self.write_to_opc_management, data_dict,
                                                                            self.model_in_use)

            #####################################################################
            ''' Get some data transformations '''
            #####################################################################
            original_col = self.orig_dict.keys()
            data_dict = Data_mgt.get_smart_features(data_dict, original_col, self.df_sta, self.json_param)

            if data_dict['new_calc'][-1]:

                #########################################
                ''' Obtain heat input recommendation '''
                #########################################

                # self.log_dict = self.get_last_quality_context_data(data_dict, context_row, self.df_sta)
                hi_rm = ES_recommendations.heat_input_recommendation(data_dict, self.results, self.json_param,
                                                                     qual_dict['last_qualities'])
                data_dict = hi_rm.recommendation_manager(self.model_in_use, data_dict['special_event'][-1],
                                                         rec_per_shaft=self.json_param['rec_per_shaft'],
                                                         qs_rec_rolling=self.json_param['qs_rec_rolling'],
                                                         rolling=self.json_param['rolling_4_rec'])

                ############################################################################
                ''' if a recommendation was done, check if recommendation is within bounds'''
                ############################################################################
                sat = hi_saturations.saturations(data_dict, self.json_param, self.df_sta)
                data_dict = sat.recommendation_vs_saturations()

                print(f"hi recommendation done after {time.time() - t0} seconds\n\n")

            else:
                print('No heat input evaluation needed')
            #####################################################################
            ''' Downtime and Transition recommendations management '''
            #####################################################################
            data_dict = ES_recommendations.downtime_recommendations(data_dict, self.json_param)

            #####################################################################
            ''' Excess air recommendations '''
            #####################################################################
            data_dict = ES_recommendations.get_stone_excess_recommendation(data_dict, qual_dict['last_qualities'],
                                                                           roll=20)

            #####################################################################
            ''' Calculate Excess air recommendation '''
            #####################################################################
            data_dict = ES_recommendations.calculate_ea_recommendations(data_dict, self.json_param, self.df_sta,
                                                                        qual_dict['filtered_co2'])

            #####################################################################
            ''' Calculate Coo ling air recommendation '''
            #####################################################################

            data_dict, send_ca_rec_mail = ES_recommendations.get_cooling_air_recommendations(data_dict, self.json_param)
            if send_ca_rec_mail:
                print('sending cooling air recommendation...')
                log_data_esc.ca_log = log_data_esc.write_cooling_air_recommendations(data_dict, self.json_param)
                title = self.mail_title_prefix + ' ' + self.kiln + ': COOLING AIR RECOMMENDATION'
                Utils.send_mail(title, log_data_esc.ca_log, self.json_param, intern=False)
            #####################################################################
            ''' Save log string and Send Mails '''
            #####################################################################

            if print_log:
                print(re.sub('  +', '', log_data_esc.log_str))

            # #####################################################################
            # ''' Extract results and store them in temporary table '''
            # #####################################################################

            param_in = [time_now, self.model_in_use, enable_opc_write, model_writing, self.kiln, self.legacy_tag_prefix]
            self.results, output_json, self.log_dict = Utils.update_results_file(self.orig_dict, data_dict,
                                                                                 self.results,
                                                                                 self.json_param, param_in)

            graph_datapoints = None
            if self.mailing & data_dict["new_calc"][-1]:
                print("Sending mail ...")
                mail_str = log_data_esc.write_complete_expert_system_log(data_dict, qual_dict, self.df_sta,
                                                                         self.json_param)

                if self.json_param["EnableMailing"] & ~data_dict['internal_loop'][-1]:
                    graph_in, intern, title_end = True, False, 'NEW QUALITY EVALUATION'
                elif data_dict['internal_loop'][-1] & (self.json_param["EnableInternalMailing"]):
                    graph_in, intern, title_end = False, True, 'RE-EVALUATION OF LAST QUALITIES'
                else:
                    title_end, graph_in, intern = None, None, None

                if title_end:

                    graph_datapoints = Utils.create_result_graph(data_dict, self.json_param['HI_units'],
                                                                 self.kiln, time_now, self.legacy_tag_prefix)

                    title = f'{self.mail_title_prefix} {self.kiln}: {title_end}'
                    str_out = Utils.send_mail(title, mail_str, self.json_param, include_graph=graph_in, intern=intern)
                    log_data_esc.log_str += str_out + "\n\n"
                    print(str_out)
                else:
                    print("sending mail is not allowed")

            if graph_datapoints:
                output_json.extend(graph_datapoints)

            output_json = json.dumps(output_json)
            print(f"exiting compute esc after {time.time() - t0} seoncds")
            return output_json

        except Exception as e:
            log_str = "problem in the principal function of the expert system controller: " + '\n' + str(e) + '\n\n'
            if print_log:
                log_save = log_data_esc.log_str
                print(f"log save during the run: \n {re.sub('  +', '', log_save)}")
            print(log_str)

            status = -1 if not enough_qual else -2

            tag_prefix1 = self.kiln + '_CC_'
            if self.legacy_tag_prefix:
                tag_prefix1 = self.kiln[:2] + '.' + self.kiln[2:4] + '.KL' + self.kiln[-1] + '.'

            output_json = {"OpcTag": tag_prefix1 + "Model_Writing_to_OPC", "Source": "Virtual",
                                "Value": status, "Timestamp": datetime.datetime.now()}
            return output_json

    def write_results(self):
        self.results.to_csv(self.filepath_results, sep=',', header='infer', encoding='utf-8', index=False)

    def run(self, cycles2run=None, print_log=True, import_data=True):

        output_json = ""
        if import_data:
            self.extract_data()

        if cycles2run is None:
            output_json = self.compute_controller_recommendations(print_log=print_log)

        else:
            self.orig_dict_backup = self.orig_dict.copy()
            self.results_backup = self.results.copy()
            for cycle in cycles2run:
                print(f"Evaluation of cycle: {str(cycle)} \n")
                if ~isinstance(self.orig_dict["Cycle - Number"], pd.Series):
                    df_temp = pd.DataFrame(self.orig_dict_backup.copy())
                else:
                    df_temp = self.orig_dict_backup.copy()

                if cycle in self.orig_dict_backup["Cycle - Number"]:
                    temp = df_temp[df_temp["Cycle - Number"] <= cycle]
                    self.orig_dict = {c: temp[c].to_numpy() for c in temp.columns}
                    if not self.results.empty:
                        result_temp = self.results
                        self.results = result_temp[result_temp['Cycle - Number'] < cycle] \
                            if 'Cycle - Number' in result_temp.columns else result_temp[result_temp["Cycle"] < cycle]
                    output_json = self.compute_controller_recommendations(print_log=print_log)

                else:
                    print("Cycle not in dataset")

        self.write_results()

        return output_json


def main_controller(kiln=None, source_directory=None):
    esc = Controller(kiln=kiln, source_dir=source_directory)
    output_json = esc.run()

    if esc.influx_param['influx2_write_cc']:
        data_json = json.loads(output_json)
        data_list = list()
        for elt in data_json:
            if 'EnableOpcWrite' in elt.keys():
                data_list.append(
                    {"OpcTag": elt['DisplayName'], "Value": elt['Value'], "timestamp": elt['Timestamp']})
                print(elt['OpcTag'])
            else:
                data_list.append({"OpcTag": elt['OpcTag'], "Value": elt['Value'], "timestamp": elt['Timestamp']})

        if len(data_list) > 0:
            data = pd.DataFrame(data_list)
            pd.DataFrame()
            data = data.pivot_table(index=['timestamp'], columns=['OpcTag'], values='Value', )
            data['Timestamp'] = pd.to_datetime(data.index, format="%Y-%m-%d %H:%M:%S")
            G_Utils.write_df_to_influxdb(json_conn=esc.influx_param, df=data, measurement='CC')

    if esc.opcua_param['opcua_write_cc']:
        Utils.write_to_opcua(json_conn=esc.opcua_param, output_json=output_json, cert_dir=esc.env3 + '\\pki\\')

    return output_json


if __name__ == "__main__":
    esc = Controller(kiln="BEAIKL1", source_dir=r'C:\Users\tabbate\Desktop\simulations')
    esc.run()

    # esc.run(cycles2run=range(202221853, 202221861))
    # esc.run(cycles2run=[202221855])
    # output_json = esc.run()

    # errors = list()
    # big_numbers = list()
    # very_big_numbers = list()
    # for k in self.orig_dict.keys():
    #     try:
    #         if (self.orig_dict[k] > 9999999).any():
    #             very_big_numbers.append(k)
    #         elif (self.orig_dict[k] > 9999).any():
    #             big_numbers.append(k)
    #     except:
    #         errors.append(k)
    # test = {k: np.float32(self.orig_dict[k]) if k in very_big_numbers else np.float32(self.orig_dict[k]) if k in big_numbers else np.float16(self.orig_dict[k]) if k not in errors else
    # self.orig_dict[k] for k in self.orig_dict.keys()}
    # t0 = time.perf_counter()
    # with open('test.pickle', 'wb') as file:
    #     pickle.dump(test, file, protocol=pickle.HIGHEST_PROTOCOL)
    # print(f'time to save history {time.perf_counter() - t0} seconds')
    #
    # t0 = time.perf_counter()
    # with open('test.pickle', 'rb') as file:
    #     a = pickle.load(file)
    # print(f'time to upload history {time.perf_counter() - t0} seconds')
