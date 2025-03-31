# IMPORTS
import os
import pandas as pd
from typing import List

# CONSTANTS
TIME_ADDITION = 2
RECOLA_TIME_STEPS = 1

# LAUNCHER CLASS
class Dataset_Manager:
    def __init__(
        self, 
        output_path: str, 
        recola_folder_path: str, 
        recola_output_file: str,
        recola_rerun: bool,
        recola_participant_count: int,
    ):
        print("Initializing DATASET-MANAGER object\n")

        # CHECK FOR OUTPUT FOLDER
        self.output_folder_check(output_path)

        # RECOLA DATASET REFORMATING
        if os.path.exists(f"{output_path}/RECOLA.csv") is True and recola_rerun is False:
            self.dataset_recola = pd.read_csv(f"{output_path}/{recola_output_file}")
            print(f"Loaded RECOLA dataset \'{output_path}/{recola_output_file}\' successfully")
        else:
            self.dataset_recola = self.recola_dataset(input_path=recola_folder_path, output_file_name=f"{output_path}/{recola_output_file}", participant_count=recola_participant_count)

    def output_folder_check(self, output_folder: str,):
        print("Starting OUPUT-FOLDER-CHECK process")
        if os.path.exists(output_folder) is False:
            os.mkdir(output_folder)
            print(f"Creating folder \'{output_folder}\', and ending OUPUT-FOLDER-CHECK process\n")
        else:
            print(f"Folder \'{output_folder}\' already exists, ending OUPUT-FOLDER-CHECK process\n")

    # RECOLA
    def recola_dataset(self, input_path: str, output_file_name: str, participant_count: int,):
        
        # STEP 1: OBTAIN LIST OF DATAFRAMES
        participant_list = self.obtain_recola_data(recola_input=input_path, participants=participant_count)

        # STEP 2: NORMALISE EACH DATAFRAME
        participant_list = self.normalise_dataframes(participants=participant_list)

        # STEP 3: COMBINE ANNOTATOR COLUMNS
        participant_list = self.combine_annotators(participants=participant_list, annotator_suffix="_x", new_header="Annotator_Arousal")
        participant_list = self.combine_annotators(participants=participant_list, annotator_suffix="_y", new_header="Annotator_Valence")

        # STEP 4: CREATE WINDOWED DATAFRAMES
        participant_list = self.create_windowed_dataframes(participants=participant_list)

        # STEP 5: REMOVE UNNEEDED COLUMNS
        participant_list = self.remove_columns(participants=participant_list)
        
        # STEP 6: COMBINE DATAFRAMES
        combined_dataframe = self.combine_dataframes(participants=participant_list)

        # STEP 7: CREATE CLASS LABELS
        finalised_dataframe = self.create_class_labels(dataframe=combined_dataframe)
        
        # STEP 8: SAVE TO FILE
        finalised_dataframe.to_csv(output_file_name, index=False)
        print(f"RECOLA dataset fully formated, saved as file \'{output_file_name}\'")

        return finalised_dataframe

    def obtain_recola_data(self, recola_input: str, participants: int):
        print("Starting OBTAIN-RECOLA-DATA process")

        # List[(Participant Name, Dataframe)]
        participant_frame_list = [] 

        if participants is None or participants > len(os.listdir(recola_input)):
            participants = len(os.listdir(recola_input))

        counter = 0
        for file in os.listdir(recola_input):
            participant_name = file.replace(".csv","")
            participant_frame = pd.read_csv(f"{recola_input}/{file}")
            participant_frame_list.append((participant_name, participant_frame))
            
            counter += 1
            if counter >= participants:
                print(f"Obtain data for {participant_name}")
                break

            print(f"Obtain data for {participant_name}")
            
        print("Process OBTAIN-RECOLA-DATA succcessfully completed\n")
        return participant_frame_list

    def normalise_dataframes(self, participants: List,):
        print("Starting NORMALISE-DATAFRAME process")

        columns_to_normalise =  list(participants[0][1].filter(regex=f'^{"ComPar"}|{"audio_speech"}|{"VIDEO"}|{"Face_detection"}|{"ECG"}|{"EDA"}|{"FF"}|{"FM"}', axis=1).columns)
        updated_participants = []
        
        for participant in participants:
            dataframe = participant[1]

            for columnName in columns_to_normalise:
                minValue = dataframe[columnName].min()
                maxValue = dataframe[columnName].max()

                minRange = -1
                maxRange = 1

                dataframe[columnName] = ((dataframe[columnName] - minValue) / (maxValue - minValue)) * (maxRange - minRange) + minRange
            
            updated_participants.append((participant[0], dataframe))
            print(f"Normalised data for {participant[0]}")
        
        print("Process NORMALISE-DATAFRAME succcessfully completed\n")
        return updated_participants

    def combine_annotators(self, participants: List, annotator_suffix: str, new_header: str,):
        print(f"Starting COMBINE-ANNOTATORS process, with annotator suffixes \'{annotator_suffix}\'")

        annotator_columns = [col for col in participants[0][1].columns if col.endswith(annotator_suffix)]
        updated_participants = []

        for participant in participants:
            dataframe = participant[1]

            annotator_values = []
            for i in range(len(dataframe)):
                sum = 0
                for annotator in annotator_columns: 
                    sum += dataframe[annotator][i]

                average = sum / len(annotator_columns)
                annotator_values.append(average)
            
            dataframe = pd.concat([dataframe, pd.Series(annotator_values, name=new_header)], axis=1)
            updated_participants.append((participant[0], dataframe))
            print(f"Annotator data combined for {participant[0]}")
       
        print(f"Process COMBINE-ANNOTATORS successfully completed, with the creation of new column \'{new_header}\'\n")
        return updated_participants

    def create_windowed_dataframes(self, participants: List,):
        print("Starting CREATE-WINDOWED-DATAFRAMES process")

        updated_participants = []

        for participant in participants:
            dataframe = participant[1]

            start_time = dataframe["time in seconds"][0]
            end_time = start_time + TIME_ADDITION
            time_step = RECOLA_TIME_STEPS
            last_time = dataframe["time in seconds"][len(dataframe)-1]


            windowed_dataframe = None
            while end_time < last_time:
                new_frame = self.obtain_window_frame(window_start=start_time, window_end=end_time, dataframe=dataframe)
                
                if windowed_dataframe is None:
                    windowed_dataframe = new_frame
                else:
                    windowed_dataframe = pd.concat([windowed_dataframe, new_frame], ignore_index=True)

                start_time += time_step
                end_time += time_step

            updated_participants.append((participant[0], windowed_dataframe))
            print(f"Windowed Dataframe for {participant[0]} completed")

        print(f"Process CREATE-WINDOWED-DATAFRAMES successfully completed\n")
        return updated_participants

    def obtain_window_frame(self, window_start: float, window_end: float, dataframe: pd.DataFrame):        
        position_list = []

        for i in range(len(dataframe)):
            if dataframe.loc[i, "time in seconds"] >= window_start and dataframe.loc[i, "time in seconds"] < window_end:
                position_list.append(i)
            elif dataframe.loc[i, "time in seconds"] >= window_end:
                break
            
        window_dataframe = dataframe.iloc[position_list]
        slice_dictionary = {}
        for column in window_dataframe:
            first_row_value = window_dataframe.iloc[0][column]
            if isinstance(first_row_value,(int, float)) and not pd.isna(first_row_value):
                window_average = window_dataframe[column].mean()
                slice_dictionary[column] = [window_average]
            else:
                slice_dictionary[column] = [first_row_value]
        
        return pd.DataFrame(slice_dictionary)

    def remove_columns(self, participants: List):
        print("Starting REMOVE-COLUMNS process")
        updated_participants = []

        for participant in participants:
            dataframe = participant[1]

            # REMOVE SPECIFIC COLUMNS
            dataframe = dataframe.drop(columns=["Unnamed: 0", "time in seconds"])
            
            for column in dataframe.columns.tolist():
                first_row_value = dataframe.iloc[0][column]
                # REMOVE COLUMNS THAT HAVE NON NUMERICAL ENTRIES
                if isinstance(first_row_value, str) or pd.isna(first_row_value):
                    dataframe = dataframe.drop(columns=[column])
                
                # REMOVE OLD ANNOTATOR FEATURES
                elif column[-2:] == "_x" or column[-2:] == "_y":
                    dataframe = dataframe.drop(columns=[column])

            updated_participants.append((participant[0], dataframe))
            print(f"Removed columns from dataframe for {participant[0]}")
        
        print(f"Process REMOVE-COLUMNS successfully completed\n")
        return  updated_participants

    def combine_dataframes(self, participants: List):
        print("Starting COMBINE-DATAFRAMES process")
        combined_dataframe = None

        for participant in participants:
            participant_number = participant[0]
            dataframe = participant[1]
            dataframe.insert(0, "Participant Number", participant_number)

            if combined_dataframe is None:
                combined_dataframe = dataframe
            else:
                combined_dataframe = pd.concat([combined_dataframe, dataframe], axis=0, ignore_index=True)
            print(f"Dataframe for {participant[0]} combined")

        print(f"Process COMBINE-DATAFRAMES successfully completed\n")
        return combined_dataframe
        
    def create_class_labels(self, dataframe: pd.DataFrame):
        print("Starting CREATE-CLASS-LABELS process")

        # AROUSAL CLASS LABEL
        class_label, values = "Class_Label_Arousal", []
        median_arousal = dataframe["Annotator_Arousal"].median()

        for index, row in dataframe.iterrows():
            if row["Annotator_Arousal"] > median_arousal:
                values.append(1)
            else:
                values.append(0)

        dataframe.insert(len(dataframe.columns)-1, class_label, values)

        # VALENCE CLASS LABEL
        class_label, values = "Class_Label_Valence", []
        median_arousal = dataframe["Annotator_Valence"].median()

        for index, row in dataframe.iterrows():
            if row["Annotator_Valence"] > median_arousal:
                values.append(1)
            else:
                values.append(0)

        dataframe.insert(len(dataframe.columns), "New_Column", values)
        print("Process CREATE-CLASS-LABELS successfully completed")

        return dataframe






    
# TEST CODE
dataset_manager = Dataset_Manager(output_path="Formatted_Datasets", recola_folder_path="Datasets/RECOLA", recola_output_file="RECOLA.csv", recola_rerun=True, recola_participant_count=4)