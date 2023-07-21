import os
import sys
import numpy as np
from datetime import datetime as dt
from new_model import Model, DANN
from optimization.new_optimizer import Optimizer


class Trainer:
    def __init__(self,
                 models_settings,
                 theta_candidates,
                 result_dir,
                 model_dir,
                 train_sets,
                 test_sets,
                 settings,
                 write
                 ):

        print("############################################")
        print("### Training Program")
        print("############################################")
        self.models_type = []
        self.models_settings = []
        for model_settings in models_settings :
            self.models_type.append(model_settings["model_type"])
            if self.models_type[-1] == "DANN" :
                self.models_settings.append(model_settings["DANN_settings"])
            else :
                self.models_settings.append(model_settings["Model_settings"])

        self.theta_candidates = theta_candidates
        self.result_dir = result_dir
        self.model_dir = model_dir
        self.train_sets = train_sets
        self.test_sets = test_sets
        self.settings = settings
        self.write = write

    def train(self):

        # Train set
        X_Trains = [train_set["data"] for train_set in self.train_sets]
        Y_Trains = [train_set["labels"] for train_set in self.train_sets]

        # Test set
        X_Tests = [test_set["data"] for test_set in self.test_sets]
        Y_Tests = [test_set["labels"] for test_set in self.test_sets]

        self.results = []
        # ---------------------------------
        # Loop over model settings
        # ---------------------------------
        for model_type,model_settings in zip(self.models_type,self.models_settings):
            print("\n--------------------------------------------")
            print("[*] Model : {} --- Preprocessing: {}".format(model_settings["model_name"], model_settings["preprocessing"]))
            print("--------------------------------------------")

            # ---------------------------------
            # Predictions Directory
            # ---------------------------------
            predictions_dir = os.path.join(self.result_dir, model_settings["model_name"])
            # create result directory if not created
            if not os.path.exists(predictions_dir):
                os.mkdir(predictions_dir)

            # ---------------------------------
            # Loop over datasets
            # ---------------------------------
            trained_models = []
            histories = []
            best_thetas = []
            Y_hat_trains, Y_hat_tests = [], []
            Y_hat_trains_decisions, Y_hat_tests_decisions = [], []
            train_times, test_times = [], []
            mu_hats_train, mu_hats_test = [], []
            for index, _ in enumerate(X_Trains):

                print("\n\tDataset : {}".format(index+1))
                print("\t----------------")

                # model_name
                trained_model_name = self.model_dir + model_settings["model_name"]

                # ---------------------------------
                # Load Model
                # ---------------------------------
                train_start = dt.now()
                print("\t[*] Loading Model")
                if model_type == "DANN" :
                    model = DANN(
                        model_settings["model_name"],
                        model_settings["input_size"],
                        model_settings["hp_lambda"],
                        model_settings["class_weights"],
                        X_Trains[index],
                        Y_Trains[index],
                        X_Tests[index],
                        case=None
                    )
                else :
                    model = Model(
                        model_settings["model_name"],
                        X_Trains[index],
                        Y_Trains[index],
                        X_Tests[index],
                        model_settings["preprocessing"],
                        model_settings["preprocessing_method"]
                    )
                # Load Trained Model
                # model = model.load(trained_model_name)

                # ---------------------------------
                # Train Model
                # ---------------------------------
                # Train model if not trained
                print("\t[*] Training Model")
                if not (model.is_trained):
                    if model_type == "DANN" :
                        model.restructure_data(model_settings["num_epochs"],model_settings["batch_size"],Y_test=Y_Tests[index])
                        history = model.fit()
                        histories.append(history)
                    else :
                        model.fit()
                trained_models.append(model)
                train_elapsed = dt.now() - train_start
                # ---------------------------------
                # Find best theta
                # ---------------------------------
                optimizer = Optimizer(
                    model = model,
                    theta_candidates = self.theta_candidates,
                    X_Train = X_Trains[index],
                    Y_Train = Y_Trains[index]
                )
                optimizer.optimize()

                optimizer_results = optimizer.get_result()
                best_theta = optimizer_results["theta"]
                best_thetas.append(best_theta)
                
                # ---------------------------------
                # Get Predictions
                # ---------------------------------
                prediction_start = dt.now()
                print("\t[*] Get Predictions")
                Y_hat_train = model.predict(X=X_Trains[index],theta=best_theta) #(optimizer_results["probas"] >= best_theta).astype(int)
                Y_hat_test = model.predict(theta=best_theta)
                Y_hat_trains.append(Y_hat_train)
                Y_hat_tests.append(Y_hat_test)

                Y_hat_train_decisions = model.decision_function(X_Trains[index], preprocess=False)
                Y_hat_test_decisions = model.decision_function()
                Y_hat_trains_decisions.append(Y_hat_train_decisions)
                Y_hat_tests_decisions.append(Y_hat_test_decisions)

                prediction_elapsed = dt.now() - prediction_start

                train_times.append(train_elapsed)
                test_times.append(prediction_elapsed)

                # ---------------------------------
                # Compute N_roi from Test
                # ---------------------------------
                print("\t[*] Compute N_roi")        
                # compute total number of test examples in ROI

                n_test_roi = len(Y_hat_test[Y_hat_test == 1])
                n_train_roi = len(Y_hat_train[Y_hat_train == 1])

                # # ---------------------------------
                # # Estimate mu
                # # ---------------------------------
                # print("\t[*] Estimate mu")        
                # gamma_roi = optimizer_results["gamma_roi"]
                # beta_roi = optimizer_results["beta_roi"]
                # mu_hat = (n_test_roi - beta_roi)/gamma_roi
                # mu_hats.append(mu_hat)

                # ---------------------------------
                # compute nu_roi, gamma_roi and beta_roi from train
                # ---------------------------------

                # get region of interest
                roi_indexes = np.argwhere(Y_hat_train == 1)
                roi_points = Y_Trains[index][roi_indexes]
                # compute nu_roi
                nu_roi = len(roi_points)

                # compute gamma_roi
                indexes = np.argwhere(roi_points == 1)

                # get signal class predictions
                signal_predictions = roi_points[indexes]
                gamma_roi = len(signal_predictions)

                # compute beta_roi
                beta_roi = nu_roi - gamma_roi

                # compute score
                mu_hat_train = (n_train_roi - beta_roi)/(gamma_roi+sys.float_info.epsilon)
                mu_hat_test = (n_test_roi - beta_roi)/(gamma_roi+sys.float_info.epsilon)
                mu_hats_train.append(mu_hat_train)
                mu_hats_test.append(mu_hat_test)

                # ---------------------------------
                # Save Predictions
                # ---------------------------------
                print("\t[*] Saving Predictions and Scores")
                # prediction file name
                prediction_name_train = os.path.join(predictions_dir, "train_"+ str(index+1) + ".predictions")
                prediction_name_test = os.path.join(predictions_dir, "test_"+ str(index+1) + ".predictions")

                # save prediction
                self.write(prediction_name_train, Y_hat_train)
                self.write(prediction_name_test, Y_hat_test)

            self.results.append({
                "trained_models": trained_models,
                "histories": histories,
                "best_thetas": best_thetas,
                "Y_hat_trains": Y_hat_trains,
                "Y_hat_tests": Y_hat_tests,
                "Y_hat_trains_decisions": Y_hat_trains_decisions,
                "Y_hat_tests_decisions": Y_hat_tests_decisions,
                "train_times": train_times,
                "test_times": test_times,
                "mu_hat_train": mu_hats_train,
                "mu_hat_test": mu_hats_test
            })

    def _compute_ROI_fields(self):
        pass

    def get_result(self):
        return self.results
