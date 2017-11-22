import numpy as np
import unittest

from bma.bma import Bma
from bma.bma_analytical import BmaAnalytical


class BmaTest(unittest.TestCase):
    def test_bma_new(self):
        bma = Bma()
        predictions = np.array([
            [0.0968007,
             - 0.000952564],
            [-0.0294904,
             - 0.043721858],
            [0.013758987,
             0.013693456]
        ])

        observations = np.array([
            0.155241936,
            0.011606597
        ])
        bma.get_prediction_weights(predictions, observations)

    def test_bma(self):
        predictions_aapl = np.array([
            0.0968007,
            - 0.000952564,
            0.025921475,
            0.023629583,
            0.00943945,
            - 0.003833022,
            0.059966963,
            0.060302157,
            0.096236214,
            0.163017198,
            0.173363462,
            0.04352599,
            0.040257841,
            0.076392964,
            0.044495806,
            0.051775813,
            0.048933543,
            0.009753156,
            - 0.042075504,
            - 0.053008806,
            - 0.048439804,
            - 0.047348227,
            - 0.058535989,
            - 0.064891607,
            - 0.074036136,
            - 0.0703495,
            - 0.048345823,
            - 0.030417271,
            - 0.035278197,
            - 0.03187995,
        ])
        predictions_msft = np.array([
            -0.0294904,
            - 0.043721858,
            - 0.039373394,
            - 0.023925241,
            0.002702497,
            0.033609267,
            0.045853075,
            0.041569609,
            0.020144451,
            - 0.008648168,
            0.00769816,
            0.006142471,
            0.005087312,
            0.003554288,
            0.010912444,
            0.009810135,
            0.005697701,
            0.009521697,
            0.023965143,
            0.036562569,
            0.052252714,
            0.07038445,
            0.09065944,
            0.099355385,
            0.085904539,
            0.061701134,
            0.004692975,
            - 0.017128672,
            - 0.025302123,
            - 0.032590929,
        ])
        predictions_bac = np.array([
            0.013758987,
            0.013693456,
            0.014606968,
            0.015620574,
            0.014063794,
            0.012808412,
            0.013967954,
            0.014468525,
            0.015946954,
            0.014774039,
            0.015589122,
            0.016799144,
            0.018725634,
            0.017996654,
            0.016645357,
            0.017475884,
            0.017471652,
            0.016414803,
            0.013938721,
            0.013959177,
            0.014056921,
            0.013802797,
            0.01273106,
            0.01165,
            0.010393225,
            0.011111591,
            0.01184893,
            0.014629405,
            0.013883818,
            0.012515113,
        ])

        observations_aapl = np.array([
            0.155241936,
            0.011606597,
            0.055099648,
            0.050091631,
            0.033816425,
            - 0.022105876,
            0.128623188,
            0.083333333,
            0.119837115,
            0.197429907,
            0.219512195,
            0.066880685,
            0.003076923,
            0.126753247,
            0.06097561,
            0.054634146,
            0.059177533,
            0.054192229,
            - 0.069156293,
            - 0.028965517,
            - 0.040240518,
            - 0.017518939,
            - 0.072744908,
            - 0.024764735,
            - 0.09469697,
            - 0.060240964,
            - 0.033253012,
            - 0.006276151,
            - 0.070086338,
            - 0.013075314,
        ])
        bma = BmaAnalytical(30)
        predictions = np.vstack([predictions_msft, predictions_bac])
        # predictions = np.vstack([predictions_aapl, predictions_msft, predictions_bac])
        bma.get_prediction_weights(predictions, observations_aapl)
        # bma = Bma()
        # bma.get_weights(predictions_aapl[0:1], predictions_msft[0:1], predictions_bac[0:1], observations_aapl[0:1])