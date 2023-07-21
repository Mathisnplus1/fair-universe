import numpy as np


class Optimizer:
    def __init__(self,
                 model,
                 theta_candidates,
                 X_Train,
                 Y_Train
                 ):

        self.model = model
        self.theta_candidates = theta_candidates
        self.X_Train = X_Train
        self.Y_Train = Y_Train

    def optimize(self):
        print("\n--------------------------------------------")
        print("[*] Model : {} ".format(self.model.model_name))
        print("--------------------------------------------")

        # ---------------------------------
        # Get scores
        # ---------------------------------
        print("\t[*] Get scores")
        if self.model.model_name[0:4] == "DANN" :
            self.probas = self.model.predict_probas(self.X_Train)[0][:,1]
        else :
            self.probas = self.model.clf.predict_proba(self.X_Train)[:,1]

        score_list = []
        nu_roi_list, beta_roi_list, gamma_roi_list = [], [], []

        for theta in self.theta_candidates :

            print("\t[*] Theta : {} ".format(theta))
            roi_mask = (self.probas >= theta)

            # ---------------------------------
            # Estimate $\nu_{ROI}$
            # ---------------------------------
            print("\t\t[*] Estimate nu_ROI")
            nu_roi = sum(roi_mask)
            nu_roi_list.append(nu_roi)

            # ---------------------------------
            # Estimate $\beta_{ROI}$
            # ---------------------------------
            print("\t\t[*] Estimate beta_ROI")
            beta_roi = sum(roi_mask & (self.Y_Train == 0)) # the number of false positive
            beta_roi_list.append(beta_roi)
            
            # ---------------------------------
            # Estimate $\gamma_{ROI}$
            # ---------------------------------
            print("\t\t[*] Estimate gamma_ROI")
            gamma_roi = sum(roi_mask & (self.Y_Train == 1)) # the number of true positive
            gamma_roi_list.append(gamma_roi)

            # score_list.append((2 * ((gamma_roi+beta_roi+10)*np.log(1+(gamma_roi/(beta_roi+10)))-gamma_roi))**0.5)
            score_list.append(self._score(nu_roi, gamma_roi))
        
        self.best_score = np.argmin(score_list)
        self.best_theta = self.theta_candidates[np.argmin(score_list)]
        self.best_nu_roi = nu_roi_list[np.argmin(score_list)]
        self.best_beta_roi = beta_roi_list[np.argmin(score_list)]
        self.best_gamma_roi = gamma_roi_list[np.argmin(score_list)]

    def _score(self, nu_roi, gamma_roi):
        """
        $\sigma^{2}_{\hat{\mu}}$ = $\frac{\nu_{ROI}}{\gamma^{2}_{ROI}}$
        """
        sigma_squared_mu_hat = nu_roi/np.square(gamma_roi)
        return sigma_squared_mu_hat

    def get_result(self):
        return {
            "probas":self.probas,
            "theta": self.best_theta,
            "score": self.best_score,
            "nu_roi": self.best_nu_roi,
            "beta_roi": self.best_beta_roi,
            "gamma_roi": self.best_gamma_roi
        }
