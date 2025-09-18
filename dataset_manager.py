# IMPORTS
import numpy as np
import os
import pandas as pd
from typing import Any, List
import random
from sklearn.preprocessing import MinMaxScaler
import subprocess

# AGAIN CONSTANTS
AGAIN_CHUNK_LENGTH = 3
AGAIN_DATA_DIRECTORY = "Datasets_Formatted/AGAIN"
AGAIN_DATASET_LOCATIONS = {
    # FULL DATASET
    "AGAIN_FULL": f"{AGAIN_DATA_DIRECTORY}/AGAIN.csv",

    # GENRE DATASETS
    "AGAIN_PLATFORMER": f"{AGAIN_DATA_DIRECTORY}/Platformer/AGAIN_Platformer.csv",
    "AGAIN_RACING": f"{AGAIN_DATA_DIRECTORY}/Racing/AGAIN_Racing.csv",
    "AGAIN_SHOOTER": f"{AGAIN_DATA_DIRECTORY}/Shooter/AGAIN_Shooter.csv",

    # PLATFORMER DATASETS
    "AGAIN_PLATFORMER_ENDLESS": f"{AGAIN_DATA_DIRECTORY}/Platformer/Games/Endless.csv",
    "AGAIN_PLATFORMER_PIRATES!": f"{AGAIN_DATA_DIRECTORY}/Platformer/Games/Pirates!.csv",
    "AGAIN_PLATFORMER_RUN'N'GUN": f"{AGAIN_DATA_DIRECTORY}/Platformer/Games/Run'N'Gun.csv",

    # RACING DATASETS
    "AGAIN_RACING_APEXSPEED": f"{AGAIN_DATA_DIRECTORY}/Racing/Games/ApexSpeed.csv",
    "AGAIN_RACING_SOLID": f"{AGAIN_DATA_DIRECTORY}/Racing/Games/Solid.csv",
    "AGAIN_RACING_TINYCARS": f"{AGAIN_DATA_DIRECTORY}/Racing/Games/TinyCars.csv",

    # SHOOTER DATASETS
    "AGAIN_SHOOTER_HEIST!": f"{AGAIN_DATA_DIRECTORY}/Shooter/Games/Heist!.csv",
    "AGAIN_SHOOTER_SHOOTOUT": f"{AGAIN_DATA_DIRECTORY}/Shooter/Games/Shootout.csv",
    "AGAIN_SHOOTER_TOPDOWN": f"{AGAIN_DATA_DIRECTORY}/Shooter/Games/TopDown.csv",
}
AGAIN_RAW_FILE = "Datasets_Raw/AGAIN/clean_data.csv"
AGAIN_TEST_DATASETS = {}
AGAIN_TEST_LOCATION = "Datasets_Test/AGAIN"

# RECOLA CONSTANTS
RECOLA_BASE_DATASET_PATH = "Datasets_Formatted/RECOLA_Base.csv"
RECOLA_CHUNK_LENGTH = 2
RECOLA_MODALITY_MAP = {
    "Audio": f"{"ComPar"}|{"audio_speech"}",
    "Video": f"{"VIDEO"}|{"Face_detection"}",
    "Physiology": f"{"ECG"}|{"EDA"}"
}
RECOLA_RAW_DATASET_DIRECTORY = "Datasets_Raw/RECOLA/Dataset_RECOLA"
RECOLA_TEST_DATASETS = {}
RECOLA_TEST_LOCATION = "Datasets_Test/RECOLA"
RECOLA_TIME_STEPS = 1
RECOLA_USER_INFO = "Datasets_Raw/RECOLA/recola_user_info.xls"


class AGAIN_Manager():
    def __init__(self, reload_datasets: bool=False):
        # STEP 0: OBTAIN LIST OF FILE PATHS
        list_of_file_paths = []
        for file_name in AGAIN_DATASET_LOCATIONS: 
            list_of_file_paths.append(AGAIN_DATASET_LOCATIONS[file_name])
        
        # STEP 1: IF RELOAD IS TRUE, DELETE OLD FILES
        if reload_datasets is True:          
            for file_path in list_of_file_paths:
                if os.path.isfile(file_path):
                    os.remove(file_path)

        # STEP 2: KEEP ONLY THE PATHS THAT ARE MISSING
        temp_list = []
        for path in list_of_file_paths:
            if not os.path.exists(path):
                temp_list.append(path)
        list_of_file_paths = temp_list
    
        # STEP 3: IF FILES DONT EXIST, CREATE DATASETS
        while list_of_file_paths:
            finished_datasets = []

            for file_path in list_of_file_paths:
                if not os.path.exists(file_path):
                    save_flag = self.__dataset_creation_manager(file_path)
                    if save_flag is True: finished_datasets.append(file_path)

            list_of_file_paths = [item for item in list_of_file_paths if item not in finished_datasets]

    def __check_files_existence(self, datasets: List) -> bool:
        check_flag = True

        for dataset in datasets:
            if not os.path.exists(AGAIN_DATASET_LOCATIONS[dataset]):
                check_flag = False
        return check_flag

    def __chunk_to_frame(self, chunk: pd.DataFrame) -> pd.DataFrame:
        column_names = list(chunk.columns)
        dataset_frame = {}
        for column in column_names:
            if column == "[control]player_id":            
                dataset_frame[column] = list(chunk[column])[0]
            elif column == "[control]game":
                dataset_frame[column] = list(chunk[column])[0]
            elif column == "[control]genre":

                dataset_frame[column] = list(chunk[column])[0]
            else:
                dataset_frame[column] = sum(chunk[column]) / float(len(chunk[column]))
        return dataset_frame

    def __create_binary_target(self, dataset: pd.DataFrame, participants: List) -> pd.DataFrame:
        # STEP 1: DECLARE CONTROL VARIABLE
        modified_dataset = []

        # STEP 2: LOOP THROUGH LIST OF PARTICIPANTS AND CREATE BINARY TARGET FEATURE
        for participant in participants:
            # STEP 2.1: GET PARTICIPANT SUBSET AND THE AROUSAL RESULTS
            participant_subset = dataset[dataset["[control]player_id"] == participant].reset_index(drop=True)
            target_results = list(participant_subset["[output]arousal"])

            # STEP 2.2: CREATE BINARY TARGET COLUMN
            if target_results:
                target_matrix = [[target] for target in target_results]
                scaler = MinMaxScaler()
                scaler.fit(target_matrix)
                
                normalised_target_results = list((scaler.transform(target_matrix)).flatten())
                median_target_value = np.median(normalised_target_results)

                binary_target = []
                for result in normalised_target_results:
                    if result > median_target_value + 0.05:
                        binary_target.append(1)
                    elif result < median_target_value - 0.05:
                        binary_target.append(0)
                    else:
                        binary_target.append(None)
                participant_subset["Binary_Arousal_Class"] = binary_target

                temp_list = participant_subset.to_dict(orient='records')
                for record in temp_list:
                    modified_dataset.append(record)

        # STEP 3: RETURN MODIFIED DATASET
        return (pd.DataFrame(modified_dataset)).dropna()
    
    def __dataset_creation_manager(self, dataset_path: str) -> bool:
        # STEP 1: SPLIT PATH INTO SECTIONS AND DECLARE CONTROL VARIABLES
        dataset_details = dataset_path.split("/")

        # STEP 2: CREATE THE DIFFERENT DATASETS
        base_dataset = None
        
        # CASE 1: FULL DATASET
        if len(dataset_details) == 3:
            # CASE 1 - STEP 1: GET LIST OF RELEVANT DATASET SUBTSETS
            genre_datasets = list(AGAIN_DATASET_LOCATIONS)[1:4]

            # CASE 1 - STEP 2: CHECKS IF DATASET CAN BE CREATED
            if self.__check_files_existence(datasets=genre_datasets) is True:
                datasets_to_combine = []
                for path in genre_datasets:
                    datasets_to_combine.append(pd.read_csv(AGAIN_DATASET_LOCATIONS[path]))
                
                genre_dataframe = pd.concat(datasets_to_combine, ignore_index=True)
                genre_dataframe.to_csv(path_or_buf=dataset_path, index=False)
                return True
            else:
                return False
        
        # CASE 2: GENRE DATASET
        elif len(dataset_details) == 4:
            # CASE 2 - STEP 1: GET LIST OF RELEVANT DATASET SUBTSETS
            game_datasets = self.__filter_games(games=list(AGAIN_DATASET_LOCATIONS)[4:], genre=dataset_details[2])
            
            # CASE 2 - STEP 2: CHECKS IF DATASET CAN BE CREATED
            if self.__check_files_existence(datasets=game_datasets) is True:
                datasets_to_combine = []
                for path in game_datasets:
                    datasets_to_combine.append(pd.read_csv(AGAIN_DATASET_LOCATIONS[path]))
                
                genre_dataframe = pd.concat(datasets_to_combine, ignore_index=True)
                genre_dataframe.to_csv(path_or_buf=dataset_path, index=False)
                return True
            else:
                return False
        
        # CASE 3: GAME DATASET
        elif len(dataset_details) == 5:
            game = dataset_details[-1].replace(".csv", "")
            base_dataset = self.__get_sub_datasets(delimiter=game)

            # CASE 3 - STEP 1: REMOVE EMPTY AND/OR UNNEEDED COLUMNS
            updated_dataset = self.__remove_columns(dataset=base_dataset)

            # CASE 3 - STEP 2: GET PARTICIPANT IDS
            participant_list = self.__get_participant_IDs(dataset=updated_dataset)

            # CASE 3 - STEP 3: CREATE WINDOWED DATASETS
            updated_dataset = self.__window_the_dataset(dataset=updated_dataset, participants=participant_list)

            # CASE 3 - STEP 4: ADD BINARY TARGET
            updated_dataset = self.__create_binary_target(dataset=updated_dataset, participants=participant_list)
        
            # CASE 3 - STEP 5: REMOVE SOME FINAL EXTRA COLUMNS
            final_dataset = updated_dataset.drop(columns=["[control]time_stamp", "[general]time_passed"])

            # CASE 3 - STEP 6: SAVE TO FILE AND RETURN SAVE FLAG
            final_dataset.to_csv(path_or_buf=dataset_path, index=False)
            return True
        
    def __filter_games(self, games: List, genre: str) -> List:
        filtered_games = []
        for game in games:
            if genre.upper() in game:
                filtered_games.append(game)
        return filtered_games

    def __get_participant_IDs(self, dataset: pd.DataFrame) -> List:
        participants_list = []
        for _, row in dataset.iterrows():
            participants_list.append(row["[control]player_id"])

        return list(set(participants_list))

    def __get_sub_datasets(self, delimiter: str) -> pd.DataFrame:
        # STEP 1: LOAD BASE DATASET
        base_dataset = pd.read_csv(AGAIN_RAW_FILE, low_memory=False)

        # STEP 2: DETERMINE WHICH COLUMN TO USE
        columns_to_search = ["[control]genre", "[control]game"]
        matching_column = None
        for column in columns_to_search:
            if base_dataset[column].eq(delimiter).any(): 
                matching_column = column
                break
        
        # STEP 3: RETURN SUB DATASET
        return base_dataset[base_dataset[matching_column] == delimiter]

    def __remove_columns(self, dataset: pd.DataFrame) -> pd.DataFrame:
        # STEP 1: DROP NA COLUMNS
        updated_dataset = dataset.dropna(axis=1)

        # STEP 2: REMOVE SPECIFIC COLUMNS
        columns_to_remove = ["[control]session_id", "[control]time_index", "[control]engine_tick", "[control]epoch"]
        for column in columns_to_remove:
            updated_dataset = updated_dataset.drop(column, axis=1)

        # STEP 3: REMOVE COLUMNS THAT ONLY CONTAIN 1 VALUE
        constant_cols = [col for col in updated_dataset.columns if updated_dataset[col].nunique(dropna=False) == 1]
        columns_to_keep = ['[control]genre', '[control]game']
        cols_to_drop = [col for col in constant_cols if col not in columns_to_keep]
        updated_dataset = updated_dataset.drop(columns=cols_to_drop)
        
        # STEP 4: RETURN MODIFIED DATASET
        return updated_dataset

    def __window_the_dataset(self, dataset: pd.DataFrame, participants: List) -> pd.DataFrame:
        # STEP 1: DECLARE CONTROL VARIABLES
        start_time = 0
        end_time = start_time + AGAIN_CHUNK_LENGTH
        dataset_frames = []

        # STEP 2: LOOP THROUGH LIST OF PARTICIPANTS AND APPLY THE WINDOWING PROCESS
        for participant in participants:
            # STEP 2.1: OBTAIN PARTICIPANT SUBSET AND FINAL TIME STAMP
            participant_subset = dataset[dataset["[control]player_id"] == participant].reset_index(drop=True)
            final_time_stamp = participant_subset.iloc[participant_subset.shape[0]-1]["[control]time_stamp"]

            # STEP 2.2: GET CHUNKS FROM SUBSET
            data_chunks_list = []
            while start_time <= final_time_stamp:
                data_chunk = participant_subset[(participant_subset['[control]time_stamp'] >= start_time) & (participant_subset['[control]time_stamp'] <= end_time)]
                
                if data_chunk.empty is False:
                    data_chunks_list.append(data_chunk)
                
                start_time += AGAIN_CHUNK_LENGTH
                end_time += AGAIN_CHUNK_LENGTH
            data_chunks_list = data_chunks_list[:-1]

            # STEP 2.3: RESET TIME COUNTERS
            start_time = 0
            end_time = start_time + AGAIN_CHUNK_LENGTH

            # STEP 2.4: COMPRESS EACH CHUNK INTO A SINGLE FRAME
            for data_chunk in data_chunks_list:
                dataset_frames.append(self.__chunk_to_frame(chunk=data_chunk))       

        # STEP 3: COMBINE INTO A SINGLE DATAFRAME AND RETURN RESULT
        windowed_dataset = pd.DataFrame(dataset_frames)
        return windowed_dataset.reset_index(drop=True)


    def add_test_dataset(self, dataset_name: str, dataset_path: str):
        AGAIN_TEST_DATASETS[dataset_name] = dataset_path

    def load_dataset(self, dataset_name: str) -> pd.DataFrame:
        return pd.read_csv(AGAIN_DATASET_LOCATIONS[dataset_name.upper()])
    
    def get_invariant_features(self, input_file: str, output_file: str, invariant_feature_count: str) -> List:
        # STEP 1: SET UP R SCRIPT CONSTANTS
        RSCRIPT = "C:/Program Files/R/R-4.4.3/bin/x64/Rscript.exe"
        INVARIANT_FEATURES_FILE = "invariant_features.R" 
        run_params = [RSCRIPT, INVARIANT_FEATURES_FILE, "inv_pred_again", input_file, output_file, invariant_feature_count]

        # STEP 2: RUN THE R SCRIPT
        try:
            subprocess.run(run_params, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e: 
            print("Error running R script:", e)

        # STEP 3: OBTAIN INVARIANT FEATURES AND CLEAN TEMPORARY FILES
        invariant_features_RECOLA = pd.read_csv(output_file)
        os.remove(input_file)
        os.remove(output_file)

        return list(invariant_features_RECOLA["invariant_feature_names"])
    
    def save_test_dataset(self, dataset_name: str, data: pd.DataFrame):
        path = f"Datasets_Test/AGAIN/{dataset_name}"
        data.to_csv(path_or_buf=path, index=False)
        self.add_test_dataset(dataset_name=dataset_name, dataset_path=path)

    # TRANSFORMATION FUNCTIONS
    def get_participant_groups(self, data: pd.DataFrame, group_size: int, group_count:int) -> pd.DataFrame:
        # Get unique player_ids
        unique_players = data["[control]player_id"].unique()
        
        total_needed = group_size * group_count
        selected_players = unique_players[:total_needed]
        
        # Map each selected player to a group ID
        participant_map = {}
        for i in range(group_count):
            group_players = selected_players[i*group_size:(i+1)*group_size]
            for pid in group_players:
                participant_map[pid] = i  # Assign group i as participant_id

        # Create a new column with participant_id, NaN for unassigned players
        data['group_id'] = data["[control]player_id"].map(participant_map)

        # Drop rows not assigned to any group
        data = data[data['group_id'].notna()].copy()
        
        return data

class RECOLA_Manager():
    def __init__(self, reload_dataset: bool = False):
        # STEP 0: IF RELOAD IS TRUE, DELETE BASE FILE
        if reload_dataset is True:
            if os.path.isfile(RECOLA_BASE_DATASET_PATH):
                os.remove(RECOLA_BASE_DATASET_PATH)

        # STEP 1: CHECK IF BASE DATASET EXISTS
        if not os.path.exists(RECOLA_BASE_DATASET_PATH):
            self.__dataset_creator_manager()
            self.add_dataset(dataset_name="RECOLA_Base", dataset_path=RECOLA_BASE_DATASET_PATH)
        else:
            self.add_dataset(dataset_name="RECOLA_Base", dataset_path=RECOLA_BASE_DATASET_PATH)

        # STEP 2: ADD ANY PRE-GENERATED TEST DATASETS
        self.__check_test_datasets()
        
    def __check_test_datasets(self):
        test_datasets = os.listdir(RECOLA_TEST_LOCATION)
        for dataset_file_name in test_datasets:
            self.add_dataset(dataset_name=dataset_file_name.replace(".csv", ""), dataset_path=f"{RECOLA_TEST_LOCATION}/{dataset_file_name}")

    def __create_class_labels(self, dataset: pd.DataFrame):
        # AROUSAL CLASS LABEL
        class_label, values = "Class_Label_Arousal", []
        median_arousal = dataset["Annotator_Arousal"].median()

        for index, row in dataset.iterrows():
            if row["Annotator_Arousal"] > median_arousal:
                values.append(1)
            else:
                values.append(0)

        dataset.insert(len(dataset.columns)-1, class_label, values)

        # VALENCE CLASS LABEL
        class_label, values = "Class_Label_Valence", []
        median_arousal = dataset["Annotator_Valence"].median()

        for index, row in dataset.iterrows():
            if row["Annotator_Valence"] > median_arousal:
                values.append(1)
            else:
                values.append(0)

        dataset.insert(len(dataset.columns), class_label, values)
        return dataset

    def __combine_dataframes(self, datasets: List):
        combined_dataframe = None

        for participant in datasets:
            participant_number = participant[0]
            dataframe = participant[1]
            dataframe.insert(0, "Participant_Number", participant_number)

            if combined_dataframe is None:
                combined_dataframe = dataframe
            else:
                combined_dataframe = pd.concat([combined_dataframe, dataframe], axis=0, ignore_index=True)
        return combined_dataframe
   
    def __combine_dataset_annotators(self, datasets: List, suffix: str, header: str) -> List:
        annotator_columns = [col for col in datasets[0][1].columns if col.endswith(suffix)]
        updated_participants = []

        for participant in datasets:
            dataframe = participant[1]

            annotator_values = []
            for i in range(len(dataframe)):
                sum = 0
                for annotator in annotator_columns: 
                    sum += dataframe[annotator][i]

                average = sum / len(annotator_columns)
                annotator_values.append(average)
            
            dataframe = pd.concat([dataframe, pd.Series(annotator_values, name=header)], axis=1)
            updated_participants.append((participant[0], dataframe))
       
        return updated_participants

    def __dataset_creator_manager(self):
        # STEP 1: LOAD DATASET PARTS
        participant_datasets = self.__get_participant_datasets()
        
        # STEP 2: NORMALISE EACH DATASET
        updated_datasets = self.__normalise_datasets(datasets=participant_datasets)
        
        # STEP 3: COMBINE ANNOTATOR COLUMNS
        updated_datasets = self.__combine_dataset_annotators(datasets=updated_datasets, suffix="_x", header="Annotator_Arousal")
        updated_datasets = self.__combine_dataset_annotators(datasets=updated_datasets, suffix="_y", header="Annotator_Valence")

        # STEP 4: CREATE WINDOWED DATASETS
        updated_datasets = self.__window_datasets(datasets=updated_datasets)

        # STEP 5: REMOVE EXTRA COLUMNS
        updated_datasets = self.__remove_columns(datasets=updated_datasets)

        # STEP 6: COMBINE DATAFRAMES
        combined_dataset = self.__combine_dataframes(datasets=updated_datasets)
        
        # STEP 7: CREATE CLASS LABELS
        updated_dataset = self.__create_class_labels(dataset=combined_dataset)

        # STEP 8: REMOVE NAN VALUES
        updated_dataset = updated_dataset.dropna()

        # STEP 9: SAVE TO FILE
        updated_dataset.to_csv(RECOLA_BASE_DATASET_PATH, index=False)

    def __get_participant_datasets(self) -> List:
        participant_datasets = []
        for participant in os.listdir(RECOLA_RAW_DATASET_DIRECTORY):
            participant_name = participant.replace(".csv","")
            dataset = pd.read_csv(f"{RECOLA_RAW_DATASET_DIRECTORY}/{participant}")
            participant_datasets.append((participant_name, dataset))
        return participant_datasets

    def __normalise_datasets(self, datasets: List) -> List:
        # STEP 1: DECLARE CONTROL VARIABLES
        columns_to_normalise =  list(datasets[0][1].filter(regex=f'^{"ComPar"}|{"audio_speech"}|{"VIDEO"}|{"Face_detection"}|{"ECG"}|{"EDA"}|{"FF"}|{"FM"}', axis=1).columns)
        min_range = -1
        max_range = 1

        # STEP 2: APPLY NORMALISATION TO EACH PARTICIPANT DATASET
        updated_datasets = []
        for participant in datasets:
            dataset = participant[1]

            for column in columns_to_normalise:
                min_value = dataset[column].min()
                max_value = dataset[column].max()
                dataset[column] = ((dataset[column] - min_value) / (max_value - min_value)) * (max_range - min_range) + min_range

            updated_datasets.append((participant[0], dataset))

        return updated_datasets

    def __obtain_window_frame(self, window_start: float, window_end: float, dataset: pd.DataFrame) -> pd.DataFrame:
        position_list = []

        for i in range(len(dataset)):
            if dataset.loc[i, "time in seconds"] >= window_start and dataset.loc[i, "time in seconds"] < window_end:
                position_list.append(i)
            elif dataset.loc[i, "time in seconds"] >= window_end:
                break

        if not position_list:  # Check if empty
            return pd.DataFrame()  # Return an empty DataFrame instead of failing
            
        window_dataframe = dataset.iloc[position_list]
        slice_dictionary = {}
        for column in window_dataframe:
                
                first_row_value = window_dataframe.iloc[0][column]
                if isinstance(first_row_value,(int, float)) and not pd.isna(first_row_value):
                    window_average = window_dataframe[column].mean()
                    slice_dictionary[column] = [window_average]
                else:
                    slice_dictionary[column] = [first_row_value]
        
        return pd.DataFrame(slice_dictionary)

    def __remove_columns(self, datasets: List):
        updated_participants = []

        for participant in datasets:
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
            
            dataframe = dataframe.dropna()
            updated_participants.append((participant[0], dataframe))        
        return  updated_participants

    def __window_datasets(self, datasets: List) -> List:
        updated_participants = []

        for participant in datasets:
            dataframe = participant[1]

            start_time = dataframe["time in seconds"][0]
            end_time = start_time + RECOLA_CHUNK_LENGTH
            time_step = RECOLA_TIME_STEPS
            last_time = dataframe["time in seconds"][len(dataframe)-1]


            windowed_dataframe = None
            while end_time < last_time:
                new_frame = self.__obtain_window_frame(window_start=start_time, window_end=end_time, dataset=dataframe)
                
                if windowed_dataframe is None:
                    windowed_dataframe = new_frame
                else:
                    windowed_dataframe = pd.concat([windowed_dataframe, new_frame], ignore_index=True)

                start_time += time_step
                end_time += time_step

            updated_participants.append((participant[0], windowed_dataframe))
        return updated_participants


    def add_dataset(self, dataset_name: str, dataset_path: str):
        RECOLA_TEST_DATASETS[dataset_name] = dataset_path

    def get_invariant_features(self, input_file: str, output_file: str, invariant_feature_count: str) -> List:
        # STEP 1: SET UP R SCRIPT CONSTANTS
        RSCRIPT = "C:/Program Files/R/R-4.4.3/bin/x64/Rscript.exe"
        INVARIANT_FEATURES_FILE = "invariant_features.R" 
        run_params = [RSCRIPT, INVARIANT_FEATURES_FILE, "inv_pred_recola", input_file, output_file, invariant_feature_count]

        # STEP 2: RUN THE R SCRIPT
        try:
            subprocess.run(run_params, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e: 
            print("Error running R script:", e)

        # STEP 3: OBTAIN INVARIANT FEATURES AND CLEAN TEMPORARY FILES
        invariant_features_RECOLA = pd.read_csv(output_file)
        os.remove(input_file)
        os.remove(output_file)

        return list(invariant_features_RECOLA["invariant_feature_names"])
    
    def load_dataset(self, dataset_name: str) -> pd.DataFrame:
        return pd.read_csv(RECOLA_TEST_DATASETS[dataset_name])

    def show_dataset_locations(self):
        print("Test Datasets:")
        for dataset in RECOLA_TEST_DATASETS:
            print(f"{dataset}:\t{RECOLA_TEST_DATASETS[dataset]}")
        print()

    def save_test_dataset(self, dataset_name: str, data: pd.DataFrame):
        path = f"Datasets_Test/RECOLA/{dataset_name}"
        data.to_csv(path_or_buf=path, index=False)
        self.add_dataset(dataset_name=dataset_name, dataset_path=path)

    # TRANSFORMATIONS
    def keep_modality(self, data: pd.DataFrame, modality: str) -> pd.DataFrame:
        for modal in RECOLA_MODALITY_MAP:
            if modal != modality:
                data = data.drop(data.filter(regex=f"^({RECOLA_MODALITY_MAP[modal]})").columns, axis=1)
        return data

    def remove_class_label(self, data: pd.DataFrame, label_to_keep: str) -> pd.DataFrame:
        to_drop = "Class_Label_Valence"
        if label_to_keep == "arousal": to_drop = "Class_Label_Arousal"
        return data.drop(columns=[to_drop])
    
    def split_by_gender(self, data: pd.DataFrame, gender: str) -> pd.DataFrame:
        gender_id = "0"
        if  gender == "male": gender_id = "1"

        participant_details = pd.read_excel(RECOLA_USER_INFO)
        gendered_users_list = []
        for _, row in participant_details.iterrows():
            if str(row["Sex"]) is gender_id:
                gendered_users_list.append(f"P{str(row["User"])[:-2]}")

        gendered_data = []
        for _, row in data.iterrows():
            if row["Participant_Number"] in gendered_users_list:
                gendered_data.append(row)

        return pd.DataFrame(gendered_data)
    
    # DEPRECATED CODE
    def simplify_environs(self, datasets: List):
        counter = 0
        environment_dataframes = []
        for dataset in datasets:
            dataset["Participant_Number"] = f"P{counter}"
            environment_dataframes.append(dataset)
            counter += 1
        
        combined_df = pd.concat(environment_dataframes, axis=0, ignore_index=True)
        return combined_df

# TESTING CODE
