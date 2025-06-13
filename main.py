import streamlit as st 
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np
import time
from scipy.stats import gamma, beta 
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Ellipse
from itertools import accumulate

st.set_page_config(page_title="Housing Modification Economic Analysis", layout="wide")
with st.spinner("Loading Housing Modification Economic Analysis Tool ....."):
    time.sleep(1)

st.title("🏠 Uganda Housing Modification for Malaria Economic Analysis")

# Useful functions
def gamma_stats(mean, std_dev, size=1000):
    """
    Generate random samples from a gamma distribution based on given mean and standard deviation parameters,
    and calculate summary statistics.

    Parameters:
    mean (float): Mean of the distribution.
    std_dev (float): Standard deviation of the distribution.
    size (int): Number of samples to generate (default is 1000).

    Returns:
    tuple: (mean_value, confidence_interval, simulated_data, shape, scale)
        - mean_value: The mean of the simulated data
        - confidence_interval: 95% confidence interval [2.5%, 97.5%]
        - simulated_data: Array of random samples
        - shape: Shape parameter (alpha) of the gamma distribution
        - scale: Scale parameter (theta) of the gamma distribution
    """
    #np.random.seed(12345)
    if mean <= 0 or std_dev <= 0:
        raise ValueError("Mean and standard deviation must be positive for gamma distribution")

    # Calculate shape (alpha) and scale (theta) parameters
    shape = (mean / std_dev) ** 2
    scale = (std_dev ** 2) / mean

    # Generate gamma distribution samples
    simulated_data = np.random.gamma(shape, scale, size=size)

    # Calculate mean and 95% confidence interval
    mean_value = np.mean(simulated_data)
    confidence_interval = np.percentile(simulated_data, [2.5, 97.5])

    return mean_value, confidence_interval, simulated_data, shape, scale

def beta_stats(mean, std_dev, num_samples=1000):
    """
    Gen. random numbers frpom a beta distribution
    """
    #np.random.seed(12345)
    if not (0 < mean < 1):
        raise ValueError("Mean must be between 0 and 1")
    if not (0 < std_dev < np.sqrt(mean * (1 - mean))):
        raise ValueError("Std. dev. must be between 0 and sqrt(mean * (1 - mean))")

    variance = std_dev ** 2
    nu = mean * (1 - mean) / variance  - 1
    alpha = mean * nu
    beta = (1 - mean) * nu
    return np.random.beta(alpha, beta, num_samples)

def calc_averted_cases(coverage, params, updated_region_data):
    """Calc cases averted using region-specific prevalence"""

    n_households = updated_region_data["households"]
    modifiable_households = n_households * updated_region_data["modifiable_homes"]

    mud_covered = modifiable_households * coverage * updated_region_data["mud"]
    brick_covered = modifiable_households * coverage * updated_region_data["bricks"]

    mud_pop = mud_covered * updated_region_data["household_size"]
    brick_pop = brick_covered * updated_region_data["household_size"]

    mud_cases = mud_pop * updated_region_data["mal_prevalence"] * params.efficacy
    brick_cases = brick_pop * updated_region_data["mal_prevalence"] * params.efficacy
    return {
        "cases_averted": mud_cases + brick_cases,
        "cases_averted_mud": mud_cases,
        "cases_averted_brick": brick_cases
    }

def sigmoid_scaleup(t, max_coverage, growth_rate, midpoint_year):
    return max_coverage / (1 + np.exp(-growth_rate * (t - midpoint_year)))

def effective_coverage(years, max_coverage, growth_rate, midpoint_year, damage_rate):
    n_years = len(years)
    coverage = sigmoid_scaleup(years, max_coverage, growth_rate, midpoint_year)
    effective = np.zeros(n_years)

    for t in range(n_years):
        for k in range(t + 1):
            new_houses = coverage[k] - coverage[k-1] if k > 0 else coverage[0]
            survival = (1 - damage_rate) ** (t - k)
            effective[t] += new_houses * survival
    return effective


def dollar_formatter(x, pos):
    return f"${x:,.0f}"

# Create a function for drawing a 95% CI ellipse around simulated values
# This function has been lifted from the Matplotlib website
def ci_ellipse(x, y, edgecolor="k"):
    """
    Draws a 95% confidence ellipse on the current matplotlib plot based on the
    given x and y data.
    Source: Matplotlib website
    Args:
        x (array-like): The x-coordinate data.
        y (array-like): The y-coordinate data.

    Returns:
        matplotlib.patches.Ellipse: The Ellipse object representing the 95% confidence ellipse.
    """
    # Calculate the mean and covariance matrix
    mean = np.array([np.mean(x), np.mean(y)])
    cov = np.cov(x, y)

    # Calculate the eigenvalues and eigenvectors of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eig(cov)

    # Calculate the angle of rotation of the ellipse
    angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])

    # Calculate the width and height of the ellipse
    width = 2 * np.sqrt(eigenvalues[0]) * np.sqrt(5.991)  # 95% confidence level
    height = 2 * np.sqrt(eigenvalues[1]) * np.sqrt(5.991)  # 95% confidence level

    # Create and add the ellipse to the plot
    ellipse = Ellipse(xy=mean, width=width, height=height,
                      angle=np.rad2deg(angle), edgecolor=edgecolor, fc='None', lw=1)
    plt.gca().add_patch(ellipse)

    return ellipse

#=============================================================================================
# Economic parameters
# Uganda premature mortality human capital approach
discount_rate = .03
min_wage_ppp = 1298                                 # Source Wikipedia
# Similar estimates using PPP Conversion factors, GDP(LCU per Int$)
human_cap = min_wage_ppp * ((1- (1+discount_rate)**(-53))/discount_rate)*(1 + discount_rate)**(-12.5)
#================================================================================

terms_conditions = """
We this tool as a service to the public. We are not responsible for, and expressly disclaim all
liability for, damages of any kind arising out of use, reference to, or reliance on
any information contained within this tool. While the information contained within
this tool is periodically updated, no guarantee is given that the information
provided in this tool is correct, complete, and up-to-date. Although this tool
may include links providing direct access to other resources, including websites,
xxx is not responsible for the accuracy or content of
information contained in these sites. Links from the hosting websites
to third-party sites do not constitute an endorsement by us of the parties
or their products and services.

The user agrees to indemnify the xxx and hold the xxx harmless from
and against any and all claims, damages and liabilities asserted by third parties
(including claims for negligence) which arise directly or indirectly from the use of
the tool.

"""
# Tabs
tabs = st.tabs(["About", "Summary Inputs", "Regional Summary", "National Summary and Notes", "Contact Us"])
# Region specific dara from 2024 Uganda Census
with tabs[0]:
    with st.expander("Terms and conditions"):
        st.write(terms_conditions)
regions = {
    "Kampala" : {
        "population": 1_797_722,        # Total region population
        "households": 529_057,          # Number of households
        "bricks": .635,                 # Proportion brick houses
        "mud": .017,                    # Proportion mud houses
        "other_wall": .348,             # Other type of wall
        "subsistence": .017,            # Proportion subsistence farming
        "pdm": .692,                    # Proportion receiving parish funds
        "mal_prevalence": .01,          # Malaria prevalence
        "pop_growth_rate": .023,        # Population growth rate
        "pop_u5": 173_460,              # Population under-5
        "mdp": .10,                     # Multidimensional poverty index
        "household_size": 2.9,          # Household size
        "modifiable_homes": .8175,      # Proportion houses modifiable
        "rdt_pos_24": 62_854            # RDT positive 2024 from DHIS-2
    },
    "Buganda" : {
        "population": 11_171_924,
        "households": 2_894_776,
        "bricks": .66,
        "mud": .09,
        "other_wall": .25,
        "subsistence": .173,
        "pdm": .227,
        "mal_prevalence": .05,              # Verify Prevalence
        "pop_growth_rate": .034,
        "pop_u5": 1_510_794,
        "mdp": .31,
        "household_size": 3.6,
        "modifiable_homes": .83,
        "rdt_pos_24": 826_809,
    },
    "Busoga" : {
        "population": 4_363_295,
        "households": 965_299,
        "bricks": .716,
        "mud": .11,
        "other_wall": .174,
        "subsistence": .38,
        "pdm": .143,
        "mal_prevalence": .21,
        "pop_growth_rate": .021,
        "pop_u5": 659_016,
        "mdp": .61,
        "household_size": 4.4,
        "modifiable_homes": .858,
        "rdt_pos_24": 1_116_021,
    },
    "Bukedi" : {
        "population": 2_372_489,
        "households": 503_727,
        "bricks": .692,
        "mud": .195,
        "other_wall": .113,
        "subsistence": .499,
        "pdm": .184,
        "mal_prevalence": .03,
        "pop_growth_rate": .024,
        "pop_u5": 377_096,
        "mdp": .78,
        "household_size": 4.7,
        "modifiable_homes": .846,
        "rdt_pos_24": 650_086,
    },
    "Bugisu" : {
        "population": 1_827_757,
        "households": 446_015,
        "bricks": .308,
        "mud": .563,
        "other_wall": .128,
        "subsistence": .412,
        "pdm": .442,
        "mal_prevalence": .05,
        "pop_growth_rate": .024,
        "pop_u5": 252_535,
        "mdp": .72,
        "household_size": 4.0,
        "modifiable_homes": .6545,
        "rdt_pos_24": 236_052,
    },
    "Sebei" : {
        "population": 377_294,
        "households": 80_679,
        "bricks": .05,
        "mud": .845,
        "other_wall": .105,
        "subsistence": .406,
        "pdm": .763,
        "mal_prevalence": .18,        # Verify Sebei prevalence
        "pop_growth_rate": .024,
        "pop_u5": 59_147,
        "mdp": .76,
        "household_size": 4.7,
        "modifiable_homes": .525,
        "rdt_pos_24": 22_944,
    },
    "Teso" : {
        "population": 2_462_387,
        "households": 489_620,
        "bricks": .877,
        "mud": .03,
        "other_wall": .092,
        "subsistence": .473,
        "pdm": .296,
        "mal_prevalence": .08,
        "pop_growth_rate": .031,
        "pop_u5": 393_460,
        "mdp": .50,
        "household_size": 4.9,
        "modifiable_homes": .878,
        "rdt_pos_24": 796_777,
    },
    "Karamoja" : {
        "population": 1_496_117,
        "households": 313_987,
        "bricks": .204,
        "mud": .613,
        "other_wall": .183,
        "subsistence": .709,
        "pdm": .195,
        "mal_prevalence": .34,
        "pop_growth_rate": .042,
        "pop_u5": 279_291,
        "mdp": .76,
        "household_size": 4.7,
        "modifiable_homes": .602,
        "rdt_pos_24": 376_778,
    },
    "Lango" : {
        "population": 2_546_116,
        "households": 575_559,
        "bricks": .822,
        "mud": .055,
        "other_wall": .123,
        "subsistence": .476,
        "pdm": .178,
        "mal_prevalence": .13,
        "pop_growth_rate": .023,
        "pop_u5": 372_642,
        "mdp": .36,
        "household_size": 4.4,
        "modifiable_homes": .911,
        "rdt_pos_24": 732_083,
    },
    "Acholi" : {
        "population": 2_044_355,
        "households": 466_128,
        "bricks": .836,
        "mud": .035,
        "other_wall": .129,
        "subsistence": .50,
        "pdm": .218,
        "mal_prevalence": .12,
        "pop_growth_rate": .032,
        "pop_u5": 306_357,
        "mdp": .69,
        "household_size": 4.3,
        "modifiable_homes": .918,
        "rdt_pos_24": 1_056_613,
    },
    "West Nile" : {
        "population": 3_316_255,
        "households": 646_361,
        "bricks": .779,
        "mud": .103,
        "other_wall": .119,
        "subsistence": .535,
        "pdm": .182,
        "mal_prevalence": .22,
        "pop_growth_rate": .039,
        "pop_u5": 510_018,
        "mdp": .76,
        "household_size": 5.1,
        "modifiable_homes": .889,
        "rdt_pos_24": 535_071,
    },
    "Madi" : {
        "population": 553_145,
        "households": 108_262,
        "bricks": .838,
        "mud": .042,
        "other_wall": .121,
        "subsistence": .473,
        "pdm": .233,
        "mal_prevalence": .22,
        "pop_growth_rate": .039,
        "pop_u5": 78_127,
        "mdp": .76,
        "household_size": 5.0,
        "modifiable_homes": .9185,
        "rdt_pos_24": 126_988,
    },
    "Bunyoro" : {
        "population": 2_792_732,
        "households": 663_258,
        "bricks": .487,
        "mud": .359,
        "other_wall": .155,
        "subsistence": .314,
        "pdm": .206,
        "mal_prevalence": .09,
        "pop_growth_rate": .033,
        "pop_u5": 454_135,
        "mdp": .42,
        "household_size": 4.2,
        "modifiable_homes": .743,
        "rdt_pos_24": 310_050,
    },
    "Tooro" : {
        "population": 2_154_161,
        "households": 504_035,
        "bricks": .303,
        "mud": .555,
        "other_wall": .142,
        "subsistence": .33,
        "pdm": .237,
        "mal_prevalence": .05,
        "pop_growth_rate": .028,
        "pop_u5": 330_935,
        "mdp": .40,
        "household_size": 4.2,
        "modifiable_homes": .6515,
        "rdt_pos_24": 246_116,
    },
    "Rwenzori" : {
        "population": 1_233_467,
        "households": 272_449,
        "bricks": .514,
        "mud": .353,
        "other_wall": .133,
        "subsistence": .367,
        "pdm": .299,
        "mal_prevalence": .01,
        "pop_growth_rate": .026,
        "pop_u5": 192_861,
        "mdp": .35,
        "household_size": 4.4,
        "modifiable_homes": .757,
        "rdt_pos_24": 102_010,
    },
    "Ankole" : {
        "population": 3_608_968,
        "households": 842_783,
        "bricks": .458,
        "mud": .376,
        "other_wall": .166,
        "subsistence": .278,
        "pdm": .308,
        "mal_prevalence": .03,
        "pop_growth_rate": .022,
        "pop_u5": 448_734,
        "mdp": .30,
        "household_size": 4.2,
        "modifiable_homes": .729,
        "rdt_pos_24": 290_263,
    },
    "Kigezi" : {
        "population": 1_787_231,
        "households": 396_918,
        "bricks": .292,
        "mud": .541,
        "other_wall": .168,
        "subsistence": .328,
        "pdm": .284,
        "mal_prevalence": .01,
        "pop_growth_rate": .026,
        "pop_u5": 220_829,
        "mdp": .49,
        "household_size": 4.2,
        "modifiable_homes": .6455,
        "rdt_pos_24": 39_743,
    },
}
param_labels = {
    "population": "Total Population",
    "households": "Number of Households",
    "bricks": "Proportion Brick Houses",
    "mud": "Proportion Mud Houses",
    "other_wall": "Other Wall Type Proportion",
    "subsistence": "Proportion Subsistence Farming",
    "pdm": "Proportion Receiving Parish Funds",
    "mal_prevalence": "Malaria Prevalence",
    "pop_growth_rate": "Population Growth Rate",
    "pop_u5": "Population Under 5",
    "mdp": "Multidimensional Poverty Index",
    "household_size": "Average Household Size",
    "modifiable_homes": "Proportion of Modifiable Homes",
    "rdt_pos_24": "RDT Positive Cases (2024)",
}

# Select region
region = st.sidebar.selectbox("Select a region:", sorted(list(regions.keys())))

ug_regions = list(regions.keys())

with st.sidebar.expander(f"Adjust Parameters for {region}"):
    region_data = regions[region]

    # Create sliders
    updated_region_data = {}
    for key, value in region_data.items():
        label = param_labels.get(key, key)
        if isinstance(value, int):
            updated_value = st.slider(
                label,
                min_value=int(value * .5),
                max_value=int(value * 1.5),
                value=value,
                step=1,
                format="%d",
            )
        elif isinstance(value, float):
            updated_value = st.slider(
                label,
                min_value=0.0,
                max_value=(value * 1.5),
                value=float(value),
                step=.01,
                format="%.3f",
            )
        else:
            updated_value = value
        
        updated_region_data[key] = updated_value


@dataclass
class SimulationInputs:
  um_mean: float = .3
  um_sd: float = .1

  severe_mean: float = .082                                         # severe malaria proportion
  severe_sd: float = .02

  severe_anemia_mean: float = .322 * severe_mean                    # Proportion severe with anemia
  severe_anemia_sd: float = severe_anemia_mean * .1

  cerebral_malaria_mean: float = .002 * severe_mean                  # Proportion cerebral malaria
  cerebral_malaria_sd: float = cerebral_malaria_mean * .1

  cerebral_anemia: float = .001 * severe_mean                       # Proportion cerebral anemia and anemia
  cerebral_anemia_sd: float = cerebral_anemia * .1
  
  neurological: float = .098 * severe_mean                          # Risk of neurological sequelae in severe malaria
  neurological_sd: float = neurological * .1
  
  deaths: float = .001268                                           # As a proportion of UM cases WMR 2024
  deaths_sd: float = .000407                                        # Calcs. from WMR

  los: float = 2.3                                                  # Snyman 2024
  los_sd: float = .1

  dw_um: float = .2078                                              # Disability weight uncomplicated malaria
  dw_um_sd: float = dw_um * .19

  dw_cerebral: float = .471
  dw_cerebral_sd: float = .047

  dw_sev_anemia: float = .149
  dw_sev_anemia_sd: float = .0149

  dw_cerebral_anemia: float = .620
  dw_cerebral_anemia_sd: float = .062

  duration_um: float = 5.1                                          # Duration uncomplicated malaria
  duration_um_sd: float = duration_um * .19

  duration_sm: float = 8.75                                         # Duration severe
  duration_sm_sd: float = .875

  duration_sm_comp: float = 11.0                                    # Duration complications

  duration_neurological: float = 10.1
  duration_neurological_sd: float = .101

  duration_severe_anemia: float = 11.0
  duration_severe_anemia_sd: float = 1.1

  duration_cerebral: float = 6.5
  duration_cerebral_sd: float = .65

  duration_sm_overall: float = duration_sm + duration_sm_comp
  duration_sm_overall_sd: float = duration_sm_overall * .19

  dw_sm: float = .471
  dw_sm_sd: float = (.550 - .411)/(3.92 * .471)

  opd_cost: float = 5.84                                            # Snyman
  opd_cost_sd: float = .4975

  ipd_cost: float = 19.77                                           # Snyman
  ipd_cost_sd: float = 2.615

  transfusion_cost: float = 86.25                                   # Inflated from $25 in 1989
  prop_sam_transfused:float = .328                                  # Proportion severe anemia transfused
  hematinics: float = 16.16                                         # Assume three month course including labs

  hh_um_cost: float = 12.65                                         # Snyman Uncomplicated societal costs
  hh_um_cost_sd: float = 1.145

  hh_sm_cost: float = 20.29                                         # Snyman complicated societal costs
  hh_sm_cost_sd: float = 2.9975

  mean_vsly: float = 5764.17                                        # Value of statistical life year
  std_vsly: float = 2041.24
  disc_lifespan: float = 28.595                                     # Discounted remaining life

  threshold: float = 139.0
  gdp_ppp: float = 2399                                             # Adjusted for purchasing power parity
  gdp_ppp_3: float = gdp_ppp * 3
sim_data = SimulationInputs()

# Convert sim_data to dictionary
sim_data_dict = asdict(sim_data)

# Sidebar sliders
with st.sidebar.expander("Global Model Parameters"):
    updated_data = {}
    for field_name, value in sim_data_dict.items():
        label = field_name.replace("_", " ").upper()

        if isinstance(value, float):
            if 0<=value<=1:
                slider_value = st.slider(
                    label,
                    min_value=0.0,
                    max_value=1.0,
                    value=float(value),
                    step=.01,
                    format="%.3f",
                )
            else:
                slider_value = st.slider(
                    label,
                    min_value=float(value * .5),
                    max_value=float(value * 1.5),
                    value=float(value),
                    step=float(value*.05),
                    format="%.3f",
                )
        elif isinstance(value, int):
            slider_value = st.slider(
                label,
                min_value=int(value * .5),
                max_value = int(value * 1.5),
                value=value,
                step=1,
            )
        else:
            slider_value = value
        
        updated_data[field_name] = slider_value

updated_sim_data = SimulationInputs(**updated_data)


class Parameters:
    def __init__(self):
        self.years = np.arange(0, 10)
        self.max_coverage = 0.80
        self.growth_rate = 1.5
        self.midpoint_year = 5.0
        self.baseline_prevalence = 0.30
        self.efficacy = 0.32
        self.efficacy_std = 0.0625
        self.daly_per_case = 1.2
        self.failure_rate = 0.1
        self.annual_repair_rate = 0.1
        self.repair_cost_fraction = 0.3
        self.discount_rate = 0.03
        self.cost_mud = {"mean": 101, "std": 2.75}
        self.cost_brick = {"mean": 150, "std": 4.25}

    def update_from_sliders(self):
        with st.sidebar.expander("🔧 Calibration Sliders"):

            self.max_coverage = st.slider(
                "Max Coverage", 0.0, 1.0, self.max_coverage,
                help="Maximum proportion of houses that can be modified"
            )

            self.growth_rate = st.slider(
                "Growth Rate", 0.1, 5.0, self.growth_rate,
                help="Controls speed of sigmoid scale-up"
            )

            self.midpoint_year = st.slider(
                "Midpoint Year", 0, 10, int(self.midpoint_year),
                help="Year at which 50% of scale-up is achieved"
            )

            self.baseline_prevalence = st.slider(
                "Baseline Malaria Prevalence", 0.0, 1.0, self.baseline_prevalence,
                help="Proportion of population with malaria before intervention"
            )

            self.efficacy = st.slider(
                "Efficacy", 0.0, 1.0, self.efficacy,
                help="Effectiveness of house modification at reducing malaria"
            )

            self.efficacy_std = st.slider(
                "Efficacy Std Dev", 0.0, 0.2, self.efficacy_std,
                help="Standard deviation around efficacy estimate"
            )

            self.daly_per_case = st.slider(
                "DALYs per Case Averted", 0.1, 10.0, self.daly_per_case,
                help="Disability-Adjusted Life Years lost per case"
            )

            self.failure_rate = st.slider(
                "Annual Screen Failure Rate", 0.0, 0.5, self.failure_rate,
                help="Annual proportion of screens that get damaged"
            )

            self.annual_repair_rate = st.slider(
                "Annual Repair Rate", 0.0, 1.0, self.annual_repair_rate,
                help="Proportion of failed screens repaired annually"
            )

            self.repair_cost_fraction = st.slider(
                "Repair Cost Fraction", 0.0, 1.0, self.repair_cost_fraction,
                help="Repair cost as a fraction of initial cost"
            )

            self.discount_rate = st.slider(
                "Discount Rate", 0.0, 0.1, self.discount_rate,
                help="Discount rate used for economic evaluation"
            )

            self.cost_mud["mean"] = st.slider(
                "Mud Wall Cost Mean", 50, 200, self.cost_mud["mean"],
                help="Mean cost for modifying a mud-walled house"
            )

            self.cost_mud["std"] = st.slider(
                "Mud Wall Cost Std Dev", 0.0, 20.0, self.cost_mud["std"],
                help="Standard deviation for mud wall cost"
            )

            self.cost_brick["mean"] = st.slider(
                "Brick Wall Cost Mean", 50, 300, self.cost_brick["mean"],
                help="Mean cost for modifying a brick-walled house"
            )

            self.cost_brick["std"] = st.slider(
                "Brick Wall Cost Std Dev", 0.0, 20.0, self.cost_brick["std"],
                help="Standard deviation for brick wall cost"
            )

params = Parameters()
params.update_from_sliders()


def run_simulations(params, regions, n_simulations=100):
    """Run simulations with stochastic efficacy, house modification, and annual repair needs"""
    start_time = time.time()
    all_results = []

    # Pre-compute sigmoid coverage across years
    coverage_all_years = sigmoid_scaleup(
        params.years, params.max_coverage, params.growth_rate, params.midpoint_year
    )

    # Calculate beta distribution parameters for efficacy
    mu = params.efficacy
    sigma = params.efficacy_std
    efficacy_alpha = ((1 - mu) / sigma**2 - 1 / mu) * mu**2
    efficacy_beta = efficacy_alpha * (1 / mu - 1)

    # Get annual repair rate from params 
    annual_repair_rate = params.annual_repair_rate  # e.g., 0.1 means 10% need repair annually

    for _ in range(n_simulations):
        # Sample costs
        cost_mud = gamma.rvs(
            a=params.cost_mud["mean"]**2 / params.cost_mud["std"]**2,
            scale=params.cost_mud["std"]**2 / params.cost_mud["mean"]
        )
        cost_brick = gamma.rvs(
            a=params.cost_brick["mean"]**2 / params.cost_brick["std"]**2,
            scale=params.cost_brick["std"]**2 / params.cost_brick["mean"]
        )

        # Sample repair cost as a fraction of initial cost (add to params)
        repair_cost_fraction = params.repair_cost_fraction  # e.g., 0.3 means repair costs 30% of new

        # Sample efficacy
        efficacy_sample = beta.rvs(efficacy_alpha, efficacy_beta)

        region_results = []
        for region_name, region_data in regions.items():
            # Sample modifiable homes proportion with normal noise
            modifiable_sample = np.random.normal(
                loc=region_data["modifiable_homes"], scale=0.05
            )
            modifiable_sample = np.clip(modifiable_sample, 0, 1)
            modifiable_hh = region_data["households"] * modifiable_sample

            avg_cost = (
                cost_mud * region_data["mud"] + cost_brick * region_data["bricks"]
            )

            # New houses modified each year
            houses_modified = np.diff(coverage_all_years, prepend=0) * modifiable_hh
           
            cumulative_houses = np.cumsum(houses_modified)

            # Track houses needing repair separately for each year
            houses_needing_repair = np.zeros(len(params.years))
            repair_costs = np.zeros(len(params.years))

            # Initialize arrays
            cases_averted = np.zeros(len(params.years))
            costs = np.zeros(len(params.years))
            effective_houses = np.zeros(len(params.years))
            undiscounted_costs = np.zeros(len(params.years))

            for t, year in enumerate(params.years):
                # Calculate houses needing repair from previous years
                discount_factor = 1/((1 + params.discount_rate)**t)

                if t > 0:
                    # Houses that were modified in previous years and now need repair
                    for prev_t in range(t):
                        # Houses from year prev_t that need repair in current year t
                        houses_from_prev_year = houses_modified[prev_t]
                        # Apply annual repair rate (compounding for houses from earlier years)
                        repair_probability = 1 - (1 - annual_repair_rate) ** (t - prev_t)
                        houses_needing_repair[t] += houses_from_prev_year * repair_probability

                # Calculate repair costs
                repair_costs[t] = houses_needing_repair[t] * avg_cost * repair_cost_fraction

                # Calculate new house modification costs
                costs[t] = (houses_modified[t] * avg_cost + repair_costs[t]) * discount_factor
                undiscounted_costs[t] = houses_modified[t] * avg_cost + repair_costs[t]

                # Calculate effective houses (newly modified + existing - deteriorated)
                # Houses that are effective are those that have been modified but don't need repair
                effective_fraction = cumulative_houses[t] - houses_needing_repair[t]
                effective_fraction = effective_fraction / modifiable_hh if modifiable_hh > 0 else 0
                effective_fraction = np.clip(effective_fraction, 0, coverage_all_years[t])

                # Calculate cases averted based on effective houses
                mud_pop = effective_fraction * modifiable_hh * region_data["mud"] * region_data["household_size"]
                brick_pop = effective_fraction * modifiable_hh * region_data["bricks"] * region_data["household_size"]
                cases_averted[t] = (mud_pop + brick_pop) * region_data["mal_prevalence"] * efficacy_sample

            region_results.append({
                "Region": region_name,
                "Cost_Per_HH": float(avg_cost),
                "Cases_Averted": cases_averted.tolist(),
                "New_Construction_Costs": (houses_modified * avg_cost).tolist(),
                "Repair_Costs": repair_costs.tolist(),
                "Total_Costs": costs.tolist(),
                "Undiscounted_Costs": undiscounted_costs.tolist(),
                "Coverage_Target": coverage_all_years.tolist(),
                "Houses_Modified_Annual": houses_modified.tolist(),
                "Houses_Modified_Cumulative": cumulative_houses.tolist(),
                "Houses_Needing_Repair": houses_needing_repair.tolist(),
                "Effective_Coverage": (effective_fraction * 100).tolist() if isinstance(effective_fraction, float) else [ef * 100 for ef in effective_fraction]
            })
       
        all_results.append(region_results)
        
    return all_results

def analyze_results(results):
    """Flatten and summarize results"""
    flat_results = []
    for sim in results:
        for region in sim:
            flat_results.append(region)

    df = pd.DataFrame(flat_results)

    summary = df.groupby('Region').agg({
        'Cost_Per_HH': ['mean', 'std'],
        'Cases_Averted': lambda x: np.mean([val[-1] for val in x]),
        'Total_Costs': lambda x: np.mean([sum(val) for val in x]),  # Changed from 'Costs' to 'Total_Costs'
        'Houses_Modified_Cumulative': lambda x: np.mean([val[-1] for val in x]),
        'Houses_Needing_Repair': lambda x: np.mean([val[-1] for val in x]),  # Added to track repair needs
        'Repair_Costs': lambda x: np.mean([sum(val) for val in x])  # Added to track repair costs
    })

    summary.columns = ['Cost_Per_HH_Mean', 'Cost_Per_HH_Std',
                       'Year5_Cases_Averted', 'Total_Cost',
                       'Year5_Houses_Modified', 'Year5_Houses_Needing_Repair',
                       'Total_Repair_Cost']

    st.write("\nRegional Summary (Year 5):")
    st.write(summary.round(2))

    # Optional: Calculate what percentage of houses need repair by the end
    if 'Houses_Needing_Repair' in df.columns and 'Houses_Modified_Cumulative' in df.columns:
        summary['Repair_Percentage'] = (summary['Year5_Houses_Needing_Repair'] /
                                       summary['Year5_Houses_Modified'] * 100).round(1)
        print("\nPercentage of Houses Needing Repair by Year 5:")
        print(summary['Repair_Percentage'])

    return df



#=============================================================================================================

def create_stats_dataframe(nested_list):
    # Initialize lists to store the statistics
    means = []
    std_devs = []
    minimums = []
    maximums = []
    severe_mal = []
    severe_mal_lower = []
    severe_mal_upper = []
    severe_anemia = []
    severe_anemia_lower = []
    severe_anemia_upper = []
    cerebral_mal = []
    cerebral_mal_lower = []
    cerebral_mal_upper = []
    cerebral_anemia = []
    cerebral_anemia_lower = []
    cerebral_anemia_upper = []
    deaths = []
    deaths_lower = []
    deaths_upper = []
    opd_cost = []
    opd_cost_lower = []
    opd_cost_upper = []
    ipd_cost = []
    ipd_cost_lower = []
    ipd_cost_upper = []
    um_hh_cost = []
    um_hh_cost_lower = []
    um_hh_cost_upper = []
    sm_hh_cost = []
    sm_hh_cost_lower = []
    sm_hh_cost_upper = []
    vsl = []
    vsl_lower = []
    vsl_upper = []
    death_human_cap = []
    death_human_cap_lower = []
    death_human_cap_upper = []

    #Cumulative values
    cum_deaths = []
    severe_mal_cum = []
    severe_anemia_cum = []
    cerebral_mal_cum = []
    cerebral_anemia_cum = []
    ipd_cost_cum = []
    opd_cost_cum = []
    um_hh_cost_cum = []
    sm_hh_cost_cum = []
    vsl_cum = []
    death_human_cap_cum = []

    # For each position/index in the inner lists (0-9)
    for index in range(10):
        # Extract all values at the current index position across all 100 lists
        values_at_index = [inner_list[index] for inner_list in nested_list]
        sev_mal = [x * beta_stats(sim_data.severe_mean, sim_data.severe_sd, 1)[0] for x in values_at_index]
        cer_mal = [x * beta_stats(sim_data.cerebral_malaria_mean, sim_data.cerebral_malaria_sd, 1)[0] for x in values_at_index]
        cer_an = [x * beta_stats(sim_data.cerebral_anemia, sim_data.cerebral_anemia_sd, 1)[0] for x in values_at_index]
        sev_anemia = [x * beta_stats(sim_data.severe_anemia_mean, sim_data.severe_anemia_sd, 1)[0] for x in values_at_index]
        deaths_at_index = [x * beta_stats(sim_data.deaths, sim_data.deaths_sd, 1)[0] for x in values_at_index]

        cum_deaths.append(deaths_at_index)
        severe_mal_cum.append(sev_mal)
        severe_anemia_cum.append(sev_anemia)
        cerebral_mal_cum.append(cer_mal)
        cerebral_anemia_cum.append(cer_an)

        vsl_at_index = [x * sim_data.disc_lifespan * gamma_stats(sim_data.mean_vsly, sim_data.std_vsly, 1)[0] for x in deaths_at_index]
        human_cap_at_index = [x * human_cap for x in deaths_at_index]
        opd_cost_at_index = [x * gamma_stats(sim_data.opd_cost, sim_data.opd_cost_sd, 1)[0] for x in values_at_index]
        ipd_cost_at_index = [x * gamma_stats(sim_data.ipd_cost, sim_data.ipd_cost_sd, 1)[0] for x in values_at_index]
        um_hh_cost_at_index = [x * gamma_stats(sim_data.hh_um_cost, sim_data.hh_um_cost_sd, 1)[0] for x in values_at_index]
        sm_hh_cost_at_index = [x * gamma_stats(sim_data.hh_sm_cost, sim_data.hh_sm_cost_sd, 1)[0] for x in sev_mal]

        ipd_cost_cum.append(ipd_cost_at_index)
        opd_cost_cum.append(opd_cost_at_index)
        um_hh_cost_cum.append(um_hh_cost_at_index)
        sm_hh_cost_cum.append(sm_hh_cost_at_index)
        vsl_cum.append(vsl_at_index)
        death_human_cap_cum.append(human_cap_at_index)

        # Calculate statistics
        means.append(np.mean(values_at_index))
        std_devs.append(np.std(values_at_index))
        minimums.append(np.min(values_at_index))
        maximums.append(np.max(values_at_index))

        severe_mal.append(np.mean(sev_mal))
        severe_mal_lower.append(np.percentile(sev_mal, 2.5))
        severe_mal_upper.append(np.percentile(sev_mal, 97.5))

        severe_anemia.append(np.mean(sev_anemia))
        severe_anemia_lower.append(np.percentile(sev_anemia, 2.5))
        severe_anemia_upper.append(np.percentile(sev_anemia, 97.5))

        cerebral_mal.append(np.mean(cer_mal))
        cerebral_mal_lower.append(np.percentile(cer_mal, 2.5))
        cerebral_mal_upper.append(np.percentile(cer_mal, 97.5))

        cerebral_anemia.append(np.mean(cer_an))
        cerebral_anemia_lower.append(np.percentile(cer_an, 2.5))
        cerebral_anemia_upper.append(np.percentile(cer_an, 97.5))

        deaths.append(np.mean(deaths_at_index))
        deaths_lower.append(np.percentile(deaths_at_index, 2.5))
        deaths_upper.append(np.percentile(deaths_at_index, 97.5))

        opd_cost.append(np.mean(opd_cost_at_index))
        opd_cost_lower.append(np.percentile(opd_cost_at_index, 2.5))
        opd_cost_upper.append(np.percentile(opd_cost_at_index, 97.5))

        ipd_cost.append(np.mean(ipd_cost_at_index))
        ipd_cost_lower.append(np.percentile(ipd_cost_at_index, 2.5))
        ipd_cost_upper.append(np.percentile(ipd_cost_at_index, 97.5))

        vsl.append(np.mean(vsl_at_index))
        vsl_lower.append(np.percentile(vsl_at_index, 2.5))
        vsl_upper.append(np.percentile(vsl_at_index, 97.5))

        death_human_cap.append(np.mean(human_cap_at_index))
        death_human_cap_lower.append(np.percentile(human_cap_at_index, 2.5))
        death_human_cap_upper.append(np.percentile(human_cap_at_index, 97.5))

        um_hh_cost.append(np.mean(um_hh_cost_at_index))
        um_hh_cost_lower.append(np.percentile(um_hh_cost_at_index, 2.5))
        um_hh_cost_upper.append(np.percentile(um_hh_cost_at_index, 97.5))

        sm_hh_cost.append(np.mean(sm_hh_cost_at_index))
        sm_hh_cost_lower.append(np.percentile(sm_hh_cost_at_index, 2.5))
        sm_hh_cost_upper.append(np.percentile(sm_hh_cost_at_index, 97.5))

    # Create cumulative values
    opd_cost_cum = [[sum(x) for x in zip(*opd_cost_cum[:i+1])] for i in range(len(opd_cost_cum))]
    opd_cost_cum_mean = [np.mean(x) for x in opd_cost_cum]
    opd_cost_cum_lower = [np.percentile(x, 2.5) for x in opd_cost_cum]
    opd_cost_cum_upper = [np.percentile(x, 97.5) for x in opd_cost_cum]

    ipd_cost_cum = [[sum(x) for x in zip(*ipd_cost_cum[:i+1])] for i in range(len(ipd_cost_cum))]
    ipd_cost_cum_mean = [np.mean(x) for x in ipd_cost_cum]
    ipd_cost_cum_lower = [np.percentile(x, 2.5) for x in ipd_cost_cum]
    ipd_cost_cum_upper = [np.percentile(x, 97.5) for x in ipd_cost_cum]

    um_hh_cost_cum = [[sum(x) for x in zip(*um_hh_cost_cum[:i+1])] for i in range(len(um_hh_cost_cum))]
    um_hh_cost_cum_mean = [np.mean(x) for x in um_hh_cost_cum]
    um_hh_cost_cum_lower = [np.percentile(x, 2.5) for x in um_hh_cost_cum]
    um_hh_cost_cum_upper = [np.percentile(x, 97.5) for x in um_hh_cost_cum]

    sm_hh_cost_cum = [[sum(x) for x in zip(*sm_hh_cost_cum[:i+1])] for i in range(len(sm_hh_cost_cum))]
    sm_hh_cost_cum_mean = [np.mean(x) for x in sm_hh_cost_cum]
    sm_hh_cost_cum_lower = [np.percentile(x, 2.5) for x in sm_hh_cost_cum]
    sm_hh_cost_cum_upper = [np.percentile(x, 97.5) for x in sm_hh_cost_cum]
    
    severe_mal_cum = [[sum(x) for x in zip(*severe_mal_cum[:i+1])] for i in range(len(severe_mal_cum))]
    severe_mal_cum_mean = [np.mean(x) for x in severe_mal_cum]
    severe_mal_cum_lower = [np.percentile(x, 2.5) for x in severe_mal_cum]
    severe_mal_cum_upper = [np.percentile(x, 97.5) for x in severe_mal_cum]

    severe_anemia_cum = [[sum(x) for x in zip(*severe_anemia_cum[:i+1])] for i in range(len(severe_anemia_cum))]
    severe_anemia_cum_mean = [np.mean(x) for x in severe_anemia_cum]
    severe_anemia_cum_lower = [np.percentile(x, 2.5) for x in severe_anemia_cum]
    severe_anemia_cum_upper = [np.percentile(x, 97.5) for x in severe_anemia_cum]

    cerebral_mal_cum = [[sum(x) for x in zip(*cerebral_mal_cum[:i+1])] for i in range(len(cerebral_mal_cum))]
    cerebral_mal_cum_mean = [np.mean(x) for x in cerebral_mal_cum]
    cerebral_mal_cum_lower = [np.percentile(x, 2.5) for x in cerebral_mal_cum]
    cerebral_mal_cum_upper = [np.percentile(x, 97.5) for x in cerebral_mal_cum]

    cerebral_anemia_cum = [[sum(x) for x in zip(*cerebral_anemia_cum[:i+1])] for i in range(len(cerebral_anemia_cum))]
    cerebral_anemia_cum_mean = [np.mean(x) for x in cerebral_anemia_cum]
    cerebral_anemia_cum_lower = [np.percentile(x, 2.5) for x in cerebral_anemia_cum]
    cerebral_anemia_cum_upper = [np.percentile(x, 97.5) for x in cerebral_anemia_cum]

    cum_deaths = [[sum(x) for x in zip(*cum_deaths[:i+1])] for i in range(len(cum_deaths))]
    cum_death_mean = [np.mean(x) for x in cum_deaths]
    cum_death_lower = [np.percentile(x, 2.5) for x in cum_deaths]
    cum_death_upper = [np.percentile(x, 97.5) for x in cum_deaths]

    vsl_cum = [[sum(x) for x in zip(*vsl_cum[:i+1])] for i in range(len(vsl_cum))]
    vsl_cum_mean = [np.mean(x) for x in vsl_cum]
    vsl_cum_lower = [np.percentile(x, 2.5) for x in vsl_cum]
    vsl_cum_upper = [np.percentile(x, 97.5) for x in vsl_cum]

    death_human_cap_cum = [[sum(x) for x in zip(*death_human_cap_cum[:i+1])] for i in range(len(death_human_cap_cum))]
    death_human_cap_cum_mean = [np.mean(x) for x in death_human_cap_cum]
    death_human_cap_cum_lower = [np.percentile(x, 2.5) for x in death_human_cap_cum]
    death_human_cap_cum_upper = [np.percentile(x, 97.5) for x in death_human_cap_cum]
    
    cum_cases = list(accumulate(means))
    cum_cases_min = list(accumulate(minimums))
    cum_cases_max = list(accumulate(maximums))

    # Create summary dataframe
    df = pd.DataFrame({
        'averted_cases': means,
        'averted_std_dev': std_devs,
        'averted_cases_min': minimums,
        'averted_cases_max': maximums,

        "cumulative_averted_cases": cum_cases,
        "cumulative_averted_cases_lower": cum_cases_min,
        "cumulative_averted_cases_upper": cum_cases_max,

        'severe_malaria': severe_mal,
        'severe_malaria_lower': severe_mal_lower,
        'severe_malaria_upper': severe_mal_upper,
        'severe_malaria_cum': severe_mal_cum_mean,
        'severe_malaria_cum_lower': severe_mal_cum_lower,
        'severe_malaria_cum_upper': severe_mal_cum_upper,

        'severe_anemia': severe_anemia,
        'severe_anemia_lower': severe_anemia_lower,
        'severe_anemia_upper': severe_anemia_upper,
        'severe_anemia_cum': severe_anemia_cum_mean,
        'severe_anemia_cum_lower': severe_anemia_cum_lower,
        'severe_anemia_cum_upper': severe_anemia_cum_upper,

        'cerebral_malaria': cerebral_mal,
        'cerebral_malaria_lower': cerebral_mal_lower,
        'cerebral_malaria_upper': cerebral_mal_upper,
        'cerebral_malaria_cum': cerebral_mal_cum_mean,
        'cerebral_malaria_cum_lower': cerebral_mal_cum_lower,
        'cerebral_malaria_cum_upper': cerebral_mal_cum_upper,

        'cerebral_anemia': cerebral_anemia,
        'cerebral_anemia_lower': cerebral_anemia_lower,
        'cerebral_anemia_upper': cerebral_anemia_upper,
        'cerebral_anemia_cum': cerebral_anemia_cum_mean,
        'cerebral_anemia_cum_lower': cerebral_anemia_cum_lower,
        'cerebral_anemia_cum_upper': cerebral_anemia_cum_upper,

        'deaths': deaths,
        'deaths_lower': deaths_lower,
        'deaths_upper': deaths_upper,
        "cum_deaths": cum_death_mean,
        "cum_deaths_lower": cum_death_lower,
        "cum_deaths_upper": cum_death_upper,

        'opd_cost': opd_cost,
        'opd_cost_lower': opd_cost_lower,
        'opd_cost_upper': opd_cost_upper,
        'opd_cost_cum': opd_cost_cum_mean,
        'opd_cost_cum_lower': opd_cost_cum_lower,
        'opd_cost_cum_upper': opd_cost_cum_upper,

        'ipd_cost': ipd_cost,
        'ipd_cost_lower': ipd_cost_lower,
        'ipd_cost_upper': ipd_cost_upper,
        'ipd_cost_cum': ipd_cost_cum_mean,
        'ipd_cost_cum_lower': ipd_cost_cum_lower,
        'ipd_cost_cum_upper': ipd_cost_cum_upper,

        'um_hh_cost': um_hh_cost,
        'um_hh_cost_lower': um_hh_cost_lower,
        'um_hh_cost_upper': um_hh_cost_upper,
        'um_hh_cost_cum': um_hh_cost_cum_mean,
        'um_hh_cost_cum_lower': um_hh_cost_cum_lower,
        'um_hh_cost_cum_upper': um_hh_cost_cum_upper,

        'sm_hh_cost': sm_hh_cost,
        'sm_hh_cost_lower': sm_hh_cost_lower,
        'sm_hh_cost_upper': sm_hh_cost_upper,
        'sm_hh_cost_cum': sm_hh_cost_cum_mean,
        'sm_hh_cost_cum_lower': sm_hh_cost_cum_lower,
        'sm_hh_cost_cum_upper': sm_hh_cost_cum_upper,

        'vsl': vsl,
        'vsl_lower': vsl_lower,
        'vsl_upper': vsl_upper,

        'vsl_cum': vsl_cum_mean,
        'vsl_cum_lower': vsl_cum_lower,
        'vsl_cum_upper': vsl_cum_upper,

        'death_human_cap': death_human_cap,
        'death_human_cap_lower': death_human_cap_lower,
        'death_human_cap_upper': death_human_cap_upper,
        'death_human_cap_cum': death_human_cap_cum_mean,
        'death_human_cap_cum_lower': death_human_cap_cum_lower,
        'death_human_cap_cum_upper': death_human_cap_cum_upper,
    })

    return df

with tabs[1]:
    np.random.seed(42)  
    with st.expander("Summary Values for Year 5"):
        results = run_simulations(params, regions, n_simulations=100)
        final_df = analyze_results(results)
    cases_averted = final_df["Cases_Averted"][final_df["Region"]==region].tolist()

    stats_df = create_stats_dataframe(cases_averted)

    y = final_df["Total_Costs"][final_df["Region"]==region].tolist()
    z = final_df["Undiscounted_Costs"][final_df["Region"]==region].tolist()
    v = final_df['Houses_Modified_Annual'][final_df['Region'] == region].tolist()
    w = final_df['Houses_Modified_Cumulative'][final_df['Region'] == region].tolist()

    tot_house_costs = [sum(item)/len(item) for item in zip(*y)]
    undisc_house_costs = [sum(item)/len(item) for item in zip(*z)]
    house_num_mod_annual = [sum(item)/len(item) for item in zip(*v)]
    house_num_mod_cum = [sum(item)/len(item) for item in zip(*w)]
    stats_df["total_house_costs"] = tot_house_costs
    stats_df["undiscounted_house_costs"] = undisc_house_costs
    stats_df["house_num_mod_annual"] = house_num_mod_annual
    stats_df["house_num_mod_cum"] = house_num_mod_cum

    with st.expander(f"View Updated Region Data for {region}:"):
        labeled_data = {}
        for key, value in updated_region_data.items():
            label = param_labels.get(key, key)
            labeled_data[label] = value
        df = pd.DataFrame.from_dict(labeled_data, orient="index", columns=["Value"])
        st.dataframe(df)


    housing_narrative = f"""The 2024 Uganda National Census estimates the population of {region} to be {updated_region_data["population"]:,.0f}, with {updated_region_data["pop_u5"]:,.0f} 
    being children under-five.
    The estimated population growth rate was {updated_region_data["pop_growth_rate"]* 100}%. 
    The number of households according to the census were {updated_region_data["households"]:,.0f}. 
    The reported positive malaria RDTs from the DHIS-2 in 2024 was {updated_region_data["rdt_pos_24"]:,.0f}. Approximately {updated_region_data["bricks"] * 100:,.1f}% houses were made of brick
    while {updated_region_data["mud"] * 100:,.1f}% were mud houses. The estimated proportion of houses that could be modified was {updated_region_data["modifiable_homes"]*100:,.1f}%.
    {updated_region_data["subsistence"] * 100:,.1f}% were engaged in subsistence agriculture.
    """
    with st.expander(f"Regional Summary for {region}"):
        st.write(housing_narrative)

with tabs[2]:
    with st.expander(f"Summary Data for {region}"):
        stats_df.index.name = "Year"
        mean_screening_costs = final_df["Cost_Per_HH"][final_df["Region"]=="Kampala"].mean()
        stats_df["cum_screening_costs"] = stats_df["house_num_mod_cum"] * mean_screening_costs
        # Health Sector Cost Savings
        stats_df["hs_savings"] = (stats_df["opd_cost"] + stats_df["ipd_cost"] + 
                                  stats_df["severe_anemia"] * updated_sim_data.transfusion_cost * updated_sim_data.prop_sam_transfused +
                                  stats_df["severe_anemia"] * updated_sim_data.hematinics)
        stats_df["hs_savings_lower"] = (stats_df["opd_cost_lower"] + stats_df["ipd_cost_lower"] + 
                                  stats_df["severe_anemia_lower"] * updated_sim_data.transfusion_cost * updated_sim_data.prop_sam_transfused +
                                  stats_df["severe_anemia_lower"] * updated_sim_data.hematinics)
        
        stats_df["hs_savings_upper"] = (stats_df["opd_cost_upper"] + stats_df["ipd_cost_upper"] + 
                                  stats_df["severe_anemia_upper"] * updated_sim_data.transfusion_cost * updated_sim_data.prop_sam_transfused +
                                  stats_df["severe_anemia_upper"] * updated_sim_data.hematinics)
        
        stats_df["hs_savings_cum"] = (stats_df["opd_cost_cum"] + stats_df["ipd_cost_cum"] + 
                                  stats_df["severe_anemia_cum"] * updated_sim_data.transfusion_cost * updated_sim_data.prop_sam_transfused +
                                  stats_df["severe_anemia_cum"] * updated_sim_data.hematinics)
        
        stats_df["hs_savings_cum_lower"] = (stats_df["opd_cost_cum_lower"] + stats_df["ipd_cost_cum_lower"] + 
                                  stats_df["severe_anemia_cum_lower"] * updated_sim_data.transfusion_cost * updated_sim_data.prop_sam_transfused +
                                  stats_df["severe_anemia_cum_lower"] * updated_sim_data.hematinics)
        
        stats_df["hs_savings_cum_upper"] = (stats_df["opd_cost_cum_upper"] + stats_df["ipd_cost_cum_upper"] + 
                                  stats_df["severe_anemia_cum_upper"] * updated_sim_data.transfusion_cost * updated_sim_data.prop_sam_transfused +
                                  stats_df["severe_anemia_cum_upper"] * updated_sim_data.hematinics)
        
        # Societal savings
        stats_df["societal_savings_no_death"] = stats_df["hs_savings"] + stats_df["um_hh_cost"] + stats_df["sm_hh_cost"]
        stats_df["societal_savings_death"] = stats_df["hs_savings"] + stats_df["um_hh_cost"] + stats_df["sm_hh_cost"] + stats_df["vsl"]
        
        stats_df["societal_savings_no_death_lower"] = stats_df["hs_savings_lower"] + stats_df["um_hh_cost_lower"] + stats_df["sm_hh_cost_lower"]
        stats_df["societal_savings_death_lower"] = stats_df["hs_savings_lower"] + stats_df["um_hh_cost_lower"] + stats_df["sm_hh_cost_lower"] + stats_df["vsl_lower"]

        stats_df["societal_savings_no_death_upper"] = stats_df["hs_savings_upper"] + stats_df["um_hh_cost_upper"] + stats_df["sm_hh_cost_upper"]
        stats_df["societal_savings_death_upper"] = stats_df["hs_savings_upper"] + stats_df["um_hh_cost_upper"] + stats_df["sm_hh_cost_upper"] + stats_df["vsl_upper"]

        stats_df["societal_savings_cum_no_death"] = stats_df["hs_savings_cum"] + stats_df["um_hh_cost_cum"] + stats_df["sm_hh_cost_cum"]
        stats_df["societal_savings_cum_death"] = stats_df["hs_savings_cum"] + stats_df["um_hh_cost_cum"] + stats_df["sm_hh_cost_cum"] + stats_df["vsl_cum"]
        
        stats_df["societal_savings_cum_no_death_lower"] = stats_df["hs_savings_cum_lower"] + stats_df["um_hh_cost_cum_lower"] + stats_df["sm_hh_cost_cum_lower"]
        stats_df["societal_savings_cum_death_lower"] = stats_df["hs_savings_cum_lower"] + stats_df["um_hh_cost_cum_lower"] + stats_df["sm_hh_cost_cum_lower"] + stats_df["vsl_cum_lower"]

        stats_df["societal_savings_cum_no_death_upper"] = stats_df["hs_savings_cum_upper"] + stats_df["um_hh_cost_cum_upper"] + stats_df["sm_hh_cost_cum_upper"]
        stats_df["societal_savings_cum_death_upper"] = stats_df["hs_savings_cum_upper"] + stats_df["um_hh_cost_cum_upper"] + stats_df["sm_hh_cost_cum_upper"] + stats_df["vsl_cum_upper"]

        stats_df["total_house_costs_cum"] = stats_df["total_house_costs"].cumsum()
        st.dataframe(stats_df.style.format("{:,.2f}"))

        st.write(f"The discounted costs of modifying all houses in {region} over 10 years is \\${stats_df["total_house_costs"].sum():,.2f}")
        st.write(f"The undiscounted costs of modifying all houses in {region} over 10 years is \\${stats_df["undiscounted_house_costs"].sum():,.2f}")

    with st.expander(f"Net Savings Societal Perspective: {region}"):
        fig, ax = plt.subplots(figsize=(10,6))
        plt.plot(stats_df.index, stats_df["societal_savings_cum_death"] - stats_df["total_house_costs_cum"], label="Mean Costs")
        plt.plot(stats_df.index, stats_df["societal_savings_cum_death_lower"] - stats_df["total_house_costs_cum"], label="Lower Costs")
        plt.plot(stats_df.index, stats_df["societal_savings_cum_death_upper"] - stats_df["total_house_costs_cum"], label="Upper Costs")
        plt.xlabel("Time in Years")
        plt.ylabel("Dollars ($)")
        plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(dollar_formatter))
        plt.title(f"Trends in estimated cumulative cost savings: {region}")
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_position(("data", 0))
        ax.spines["left"].set_position(("data", 0))
        plt.legend(loc="best")
        plt.grid(axis="both", alpha=.2)
        st.pyplot(fig)

    with st.expander(f"Annual Economic Savings from Averted Deaths: {region}"):
        fig, ax = plt.subplots(figsize=(10, 6))
        plt.plot(stats_df.index, stats_df["vsl"], label="Mean")
        plt.fill_between(stats_df.index, stats_df["vsl_lower"], stats_df["vsl_upper"],
                        color='g', alpha=0.1, zorder=-2)
        plt.grid(axis="both", alpha=.2)
        plt.xlabel("Time in Years")
        plt.ylabel("Economic Costs ($)")
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        plt.title(f"Annual Economic Savings - Deaths Averted \n{region}", fontsize=18)
        plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(dollar_formatter))
        st.pyplot(fig)

    with st.expander(f"Cumulative Savings from Deaths Averted: {region}"):
        fig, ax = plt.subplots(figsize=(10, 6))
        plt.plot(stats_df.index, stats_df["vsl_cum"], label="Mean")
        plt.grid(axis="both", alpha=.2)
        plt.fill_between(stats_df.index, stats_df["vsl_cum_lower"], stats_df["vsl_cum_upper"],
                        color='g', alpha=0.1, zorder=-2)
        plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(dollar_formatter))
        plt.title(f"Cumulative Economic savings - Deaths Averted \n{region}", fontsize=18)
        st.pyplot(fig)

    with st.expander(f"Disability Adjusted Life Years Averted: {region}"):
        qaly_df = pd.DataFrame()

        qaly_df["death_daly_mean"] = stats_df["deaths"] * sim_data.disc_lifespan
        qaly_df["death_daly_lower"] = stats_df["deaths_lower"] * sim_data.disc_lifespan
        qaly_df["death_daly_upper"] = stats_df["deaths_upper"] * sim_data.disc_lifespan

        qaly_df["uncomp_daly_mean"] = stats_df["averted_cases"] * gamma.rvs(sim_data.duration_um, sim_data.duration_um_sd) * sim_data.dw_um/365
        qaly_df["uncomp_daly_lower"] = stats_df["averted_cases_min"] * gamma.rvs(sim_data.duration_um, sim_data.duration_um_sd) * sim_data.dw_um/365
        qaly_df["uncomp_daly_upper"] = stats_df["averted_cases_max"] * gamma.rvs(sim_data.duration_um, sim_data.duration_um_sd) * sim_data.dw_um/365
        
        qaly_df["severe_mal_daly_mean"] = stats_df["severe_malaria"] * gamma.rvs(sim_data.duration_sm, sim_data.duration_sm_sd) * beta.rvs(sim_data.dw_sm, sim_data.dw_sm_sd)/365
        qaly_df["severe_mal_daly_lower"] = stats_df["severe_malaria_lower"] * gamma.rvs(sim_data.duration_sm, sim_data.duration_sm_sd) * beta.rvs(sim_data.dw_sm, sim_data.dw_sm_sd)/365
        qaly_df["severe_mal_daly_upper"] = stats_df["severe_malaria_upper"] * gamma.rvs(sim_data.duration_sm, sim_data.duration_sm_sd) * beta.rvs(sim_data.dw_sm, sim_data.dw_sm_sd)/365
        
        qaly_df["severe_anemia_daly_mean"] = stats_df["severe_anemia"] * gamma.rvs(sim_data.duration_severe_anemia, sim_data.duration_severe_anemia_sd) * beta.rvs(sim_data.dw_sev_anemia, sim_data.dw_sev_anemia_sd)/365
        qaly_df["severe_anemia_daly_lower"] = stats_df["severe_anemia_lower"] * gamma.rvs(sim_data.duration_severe_anemia, sim_data.duration_severe_anemia_sd) * beta.rvs(sim_data.dw_sev_anemia, sim_data.dw_sev_anemia_sd)/365
        qaly_df["severe_anemia_daly_upper"] = stats_df["severe_anemia_upper"] * gamma.rvs(sim_data.duration_severe_anemia, sim_data.duration_severe_anemia_sd) * beta.rvs(sim_data.dw_sev_anemia, sim_data.dw_sev_anemia_sd)/365
        
        qaly_df["cerebral_daly_mean"] = stats_df["cerebral_malaria"] * gamma.rvs(sim_data.duration_cerebral, sim_data.duration_cerebral_sd) * beta.rvs(sim_data.dw_cerebral, sim_data.dw_cerebral_sd)/365
        qaly_df["cerebral_daly_lower"] = stats_df["cerebral_malaria_lower"] * gamma.rvs(sim_data.duration_cerebral, sim_data.duration_cerebral_sd) * beta.rvs(sim_data.dw_cerebral, sim_data.dw_cerebral_sd)/365
        qaly_df["cerebral_daly_upper"] = stats_df["cerebral_malaria_upper"] * gamma.rvs(sim_data.duration_cerebral, sim_data.duration_cerebral_sd) * beta.rvs(sim_data.dw_cerebral, sim_data.dw_cerebral_sd)/365
        
        qaly_df["cerebral_anemia_daly_mean"] = stats_df["cerebral_anemia"] * gamma.rvs(sim_data.duration_cerebral, sim_data.duration_cerebral_sd) * beta.rvs(sim_data.dw_cerebral_anemia, sim_data.dw_cerebral_anemia_sd)/365
        qaly_df["cerebral_anemia_daly_lower"] = stats_df["cerebral_anemia_lower"] * gamma.rvs(sim_data.duration_cerebral, sim_data.duration_cerebral_sd) * beta.rvs(sim_data.dw_cerebral_anemia, sim_data.dw_cerebral_anemia_sd)/365
        qaly_df["cerebral_anemia_daly_upper"] = stats_df["cerebral_anemia_upper"] * gamma.rvs(sim_data.duration_cerebral, sim_data.duration_cerebral_sd) * beta.rvs(sim_data.dw_cerebral_anemia, sim_data.dw_cerebral_anemia_sd)/365
        
        qaly_df["total_daly_averted_mean"] = (qaly_df["cerebral_daly_mean"] + qaly_df["uncomp_daly_mean"] + 
                                              qaly_df["severe_anemia_daly_mean"] + qaly_df["cerebral_anemia_daly_mean"] +
                                              qaly_df["death_daly_mean"] + qaly_df["severe_mal_daly_mean"])
        
        qaly_df["total_daly_averted_lower"] = (qaly_df["cerebral_daly_lower"] + qaly_df["uncomp_daly_lower"] + 
                                              qaly_df["severe_anemia_daly_lower"] + qaly_df["cerebral_anemia_daly_lower"] +
                                              qaly_df["death_daly_lower"] + qaly_df["severe_mal_daly_lower"])
        
        qaly_df["total_daly_averted_upper"] = (qaly_df["cerebral_daly_upper"] + qaly_df["uncomp_daly_upper"] + 
                                              qaly_df["severe_anemia_daly_upper"] + qaly_df["cerebral_anemia_daly_upper"] +
                                              qaly_df["death_daly_upper"] + qaly_df["severe_mal_daly_upper"])
        
        qaly_df["total_daly_averted_cum_lower"] = qaly_df["total_daly_averted_lower"].cumsum()
        qaly_df["total_daly_averted_cum_mean"] = qaly_df["total_daly_averted_mean"].cumsum()
        qaly_df["total_daly_averted_cum_upper"] = qaly_df["total_daly_averted_upper"].cumsum()

        st.dataframe(qaly_df.style.format("{:,.1f}"))

    averted_ip_costs = stats_df[["ipd_cost", "ipd_cost_lower", "ipd_cost_upper", "ipd_cost_cum", "ipd_cost_cum_lower", "ipd_cost_cum_upper"]][:]

    with st.expander(f"Averted Inpatient Costs for {region}"):
        averted_ip_costs.index.name = "Year"
        st.dataframe(averted_ip_costs.style.format("${:,.2f}"))
        ipd_cost_cum = averted_ip_costs["ipd_cost_cum"][-1:].values[0]
        ipd_cost_min = averted_ip_costs["ipd_cost_cum_lower"][-1:].values[0]
        ipd_cost_max = averted_ip_costs["ipd_cost_cum_upper"][-1:].values[0]
        st.write(f"Cumulative averted inpatient costs in 10 years is \\${ipd_cost_cum:,.0f} [\\${ipd_cost_min:,.0f}: \\${ipd_cost_max:,.0f}]")

    with st.expander(f"Severe Malaria Anemia Averted Costs for {region}"):
        st.write("32.8% of severe malaria anemia patients get blood transfusion. [Ackerman et. al (2021)](https://pmc.ncbi.nlm.nih.gov/articles/PMC7611367/)")
        sma_df = stats_df[["severe_anemia", "severe_anemia_lower", "severe_anemia_upper", "severe_anemia_cum", "severe_anemia_cum_lower", "severe_anemia_cum_upper"]]
        for col in sma_df.columns.tolist():
            sma_df[f"{col}_costs"] = sma_df[col] * updated_sim_data.transfusion_cost
        st.dataframe(sma_df)
        transfusion_costs_mean = sma_df["severe_anemia_cum_costs"][-1:].values[0]
        transfusion_costs_lower = sma_df["severe_anemia_cum_lower_costs"][-1:].values[0]
        transfusion_costs_upper = sma_df["severe_anemia_cum_upper_costs"][-1:].values[0]
    
        st.write(f"""The averted cumulative costs of treating severe malaria anemia in {region } in 10 years is \\${transfusion_costs_mean:,.0f} 
                 [\\${transfusion_costs_lower:,.0f}: \\${transfusion_costs_upper:,.0f}]. This assumes approximately {sma_df["severe_anemia_cum"][-1:].values[0]:,.0f} 
                 severe malaria anemia cases will be averted in that timeframe. We assume that {updated_sim_data.prop_sam_transfused * 100:,.1f}% of these cases willget blood transfusions.
                 The average cost of a blood transfusion is estimated at \\${updated_sim_data.transfusion_cost:,.0f}.""")

    with st.expander(f"Cost effectiveness acceptability curves - societal perspective: {region}"):
        st.latex(r"""
                    NMB = (E * \lambda) - C
                """)
        mean_daly = qaly_df["total_daly_averted_cum_mean"][-1:].values[0]
        min_daly = qaly_df["total_daly_averted_cum_lower"][-1:].values[0]
        max_daly = qaly_df["total_daly_averted_cum_upper"][-1:].values[0]
        sd_daly = (max_daly - min_daly)/(2 * 1.92)
        mean_costs_averted = stats_df["societal_savings_cum_death"][-1:].values[0]
        min_costs_averted = stats_df["societal_savings_cum_death_lower"][-1:].values[0]
        max_costs_averted = stats_df["societal_savings_cum_death_upper"][-1:].values[0]
        mean_screen_costs = stats_df["total_house_costs_cum"][-1:].values[0]

        mean_net_savings = (mean_costs_averted - mean_screen_costs)
        min_net_savings = (min_costs_averted - mean_screen_costs)
        max_net_savings = (max_costs_averted - mean_screen_costs)
        sd_net_savings = (max_net_savings - min_net_savings)/(2 * 1.96)

        n_sim = 10_000
        np.random.seed(12345)
        dalys_averted = np.random.normal(mean_daly, sd_daly, n_sim)
        cost_savings = np.random.normal(mean_net_savings, sd_net_savings, size=n_sim)
        
        nmb_mean = (mean_daly * updated_sim_data.threshold) - (mean_screen_costs - mean_costs_averted)
        nmb_lower = (min_daly * updated_sim_data.threshold) - (mean_screen_costs - min_costs_averted)
        nmb_upper = (max_daly * updated_sim_data.threshold) - (mean_screen_costs - max_costs_averted)

        x1 = (mean_screen_costs - mean_costs_averted)
        x2 = (mean_screen_costs - min_costs_averted)
        x3 = (mean_screen_costs - max_costs_averted)
        sd_x = (x3-x2)/(2 * 1.96)
        sd_x = abs(sd_x)
        costs_x = np.random.normal(x1, sd_x, n_sim)

        ceacs = []
        proportions = []
        
        wtp = list(range(1, 10_001))
        for t in wtp:
            props = [x/y if y !=0 else None for x,y in zip(costs_x, dalys_averted)]
            proportions.append(props)
            cutoffs = np.mean([x < t for x in props])
            ceacs.append(cutoffs)
        ceacs.sort()
        target_prob = .5
        wtp_at_50 = np.interp(target_prob, ceacs, wtp)
        prop_ce_base = np.mean(np.array(proportions) < updated_sim_data.threshold)
        prop_ce_gdp = np.mean(np.array(proportions) < updated_sim_data.gdp_ppp)
        prop_ce_gdp3 = np.mean(np.array(proportions) < updated_sim_data.gdp_ppp_3)
        if wtp_at_50 < 8_100:
            fig, ax = plt.subplots(figsize=(10,6))
            plt.plot(wtp, ceacs)
            plt.suptitle(f"Cost-effectiveness acceptability curve: {region}")
            plt.title("Societal Perspective")
            plt.ylabel("Probability cost-effective")
            plt.xlabel("Willingness-To-Pay")
            plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(dollar_formatter))
            for sp in ["top", "right"]:
                ax.spines[sp].set_visible(False)
            plt.grid(axis="both", alpha=.4, linewidth=.2)
            xlims = ax.get_xlim()
            ylims = ax.get_ylim()
            x_pos = xlims[0] + .7 * (xlims[1] - xlims[0])
            y_pos = ylims[0] + .8 * (ylims[1] - ylims[0])
            plt.annotate(f"WTP at 50% prob.: ${wtp_at_50:,.0f}",
                         xy=(x_pos, y_pos))
            st.pyplot(fig)

            st.write(f"""The net monetary benefit (societal perspective) in {region} is \\${nmb_mean:,.2f}[\\${nmb_lower:,.2f}, \\${nmb_upper:,.2f}]. 
                    A positive NMB implies the intervention is cost-effective and might be worth pursuing given competing priorities. 
                    This assumes a conservative cost-effectiveness threshold of \\${updated_sim_data.threshold}. 
                    There is a {100*prop_ce_base:,.1f}% chance the intervention is cost-effective at a CE threshold of \\${updated_sim_data.threshold:,.0f},
                    {100 * prop_ce_gdp:,.1f}% at a CET of \\${updated_sim_data.gdp_ppp}, and {100 * prop_ce_gdp3:,.1f}% at a
                    CET of \\${updated_sim_data.gdp_ppp_3}. See also the cost-effectiveness analysis plane below.""")
        
        else:
            st.write(f"""Housing screening in {region} is not cost-effective from a societal perspective using 
                     conventional cost-effectiveness thresholds including the controversial 3X WHO GDP threshold.
                     Cost-effectiveness acceptability curves for {region} are therefore not produced.
                     The net monetary benefit (societal perspective) in {region} is \\${nmb_mean:,.2f}[\\${nmb_lower:,.2f}, \\${nmb_upper:,.2f}]. 
                    A negative NMB implies the intervention is not cost-effective. 
                    This assumes a cost-effectiveness threshold of \\${updated_sim_data.threshold}.""")
            
    with st.expander(f"Cost-effectiveness acceptability curves health sector perspective: {region}"):
        # Health System Perspective
        mean_costs_averted_hs = mean_screen_costs - stats_df["hs_savings_cum"][-1:].values[0]
        min_costs_averted_hs = mean_screen_costs - stats_df["hs_savings_cum_upper"][-1:].values[0]
        max_costs_averted_hs = mean_screen_costs - stats_df["hs_savings_cum_lower"][-1:].values[0]

        sd_x_hs = (max_costs_averted_hs - min_costs_averted_hs)/(2 * 1.96)
        sd_x_hs = abs(sd_x_hs)
        costs_x_hs = np.random.normal(mean_costs_averted_hs, sd_x_hs, n_sim)
        
        ceacs = []
        proportions = []
        wtp = list(range(1, 10_001))
        for t in wtp:
            props = [x/y if y!=0 else None for x,y in zip(costs_x_hs, dalys_averted)]
            proportions.append(props)
            cutoffs = np.mean([x < t for x in props])
            ceacs.append(cutoffs)

        ceacs.sort()
        target_prob = .5
        wtp_at_50 = np.interp(target_prob, ceacs, wtp)
        if wtp_at_50 < 8_000:
            fig, ax = plt.subplots(figsize=(10,6))
            plt.plot(wtp, ceacs, label="CEAC")
            plt.suptitle(f"Cost-effectiveness acceptability curve: {region}")
            plt.title("Health system perspective")
            plt.ylabel("Probability cost-effective")
            plt.xlabel("Willingness-to-pay ($)")
            plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(dollar_formatter))
            for sp in ["top", "right"]:
                ax.spines[sp].set_visible(False)
            plt.grid(axis= "both", alpha=.4, linewidth=.2)
            plt.legend(loc="center right")
            plt.hlines(y=target_prob, xmin=0, xmax=wtp_at_50,color="red", linestyle="--", alpha=.7)
            plt.axvline(x=wtp_at_50, ymin=0.05, ymax=target_prob/plt.ylim()[1], 
            color='red', linestyle='--', alpha=0.7)
            ax.spines["bottom"].set_position(("data", 0))
            ax.spines["left"].set_position(("data", 0))
            ax.plot(0, 1.01, "^k", transform=ax.get_xaxis_transform(), clip_on=False)
            ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False)

            plt.annotate(f'WTP at 50% prob. = ${wtp_at_50:,.0f}',
                        xy=(wtp_at_50, target_prob),xytext=(15,10),
                        textcoords="offset points",)
            st.pyplot(fig)
        else:
            st.write(f"""Housing screening in {region} is not cost-effective from a healthcare perspective using 
                     conventional cost-effectiveness thresholds including the controversial 3X WHO GDP threshold.
                     Cost-effectiveness acceptability curves for {region} are therefore not produced.""")

    with st.expander(f"Cost-effectiveness plane -- societal perspective: {region}"):
        fig, ax = plt.subplots(figsize=(10,6))
        plt.scatter(dalys_averted, costs_x, s=0.5, color="g", alpha=.5)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_position(("data", 0))
        ax.spines["left"].set_position(("data", 0))
        upper_limit = np.max(dalys_averted) * updated_sim_data.threshold
        upper_daly = np.max(dalys_averted)
        lower_limit = np.min(dalys_averted) * updated_sim_data.threshold
        lower_daly = np.min(dalys_averted)
        upper_limit_gdp = upper_daly * updated_sim_data.gdp_ppp
        lower_limit_gdp = lower_daly * updated_sim_data.gdp_ppp
        upper_limit_gdp3 = upper_daly * updated_sim_data.gdp_ppp_3
        lower_limit_gdp3 = lower_daly * updated_sim_data.gdp_ppp_3
        ax.plot([lower_daly, upper_daly], [lower_limit, upper_limit], 
                label=f"CET: ${updated_sim_data.threshold:,.0f}", ls="--", color="blue")
        ax.plot([lower_daly, upper_daly], [lower_limit_gdp, upper_limit_gdp], 
                label=f"CET: ${updated_sim_data.gdp_ppp:,.0f}", ls="--", color="red")
        ax.plot([lower_daly, upper_daly], [lower_limit_gdp3, upper_limit_gdp3], 
                label=f"CET: ${updated_sim_data.gdp_ppp_3:,.0f}", ls="--", color="k")
        plt.grid(axis="both", alpha=.2, linewidth=.3)
        plt.legend(loc="lower right")
        plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(dollar_formatter))
        plt.suptitle(f"Cost-effectiveness Plane: {region}")
        plt.title("Societal Perspective")
        ci_ellipse(dalys_averted, costs_x)
        ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False)
        ax.plot(0, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False)
        plt.ylabel(r"$\Delta$ Cost", fontweight="bold")
        plt.xlabel(r"$\Delta$ DALYs", fontweight="bold")
        st.pyplot(fig)

    with st.expander(f"Cost-effectiveness plane -- health system perspective: {region}"):
        fig, ax = plt.subplots(figsize=(10,6))
        plt.scatter(dalys_averted, costs_x_hs, s=0.5, color="g", alpha=.5)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_position(("data", 0))
        ax.spines["left"].set_position(("data", 0))
        upper_limit = np.max(dalys_averted) * updated_sim_data.threshold
        upper_daly = np.max(dalys_averted)
        lower_limit = np.min(dalys_averted) * updated_sim_data.threshold
        lower_daly = np.min(dalys_averted)
        upper_limit_gdp = upper_daly * updated_sim_data.gdp_ppp
        lower_limit_gdp = lower_daly * updated_sim_data.gdp_ppp
        upper_limit_gdp3 = upper_daly * updated_sim_data.gdp_ppp_3
        lower_limit_gdp3 = lower_daly * updated_sim_data.gdp_ppp_3
        ax.plot([lower_daly, upper_daly], [lower_limit, upper_limit], 
                label=f"CET: ${updated_sim_data.threshold:,.0f}", ls="--", color="blue")
        ax.plot([lower_daly, upper_daly], [lower_limit_gdp, upper_limit_gdp], 
                label=f"CET: ${updated_sim_data.gdp_ppp:,.0f}", ls="--", color="red")
        ax.plot([lower_daly, upper_daly], [lower_limit_gdp3, upper_limit_gdp3], 
                label=f"CET: ${updated_sim_data.gdp_ppp_3:,.0f}", ls="--", color="k")
        plt.grid(axis="both", alpha=.2, linewidth=.3)
        plt.legend(loc="lower right")
        plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(dollar_formatter))
        plt.suptitle(f"Cost-effectiveness Plane: {region}")
        plt.title("Health System Perspective")
        ci_ellipse(dalys_averted, costs_x_hs)
        ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False)
        ax.plot(0, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False)
        plt.ylabel(r"$\Delta$ Cost", fontweight="bold")
        plt.xlabel(r"$\Delta$ DALYs", fontweight="bold")
        st.pyplot(fig)

with tabs[3]:
    with st.expander("View Combined National Level Data"):
        st.dataframe(final_df)
        
    with st.expander("Additional Notes"):
        additional_notes = f"""
        1. 32.8% of severe malaria anemia patients get blood transfusion. [Ackerman et. al (2021)](https://pmc.ncbi.nlm.nih.gov/articles/PMC7611367/)
        2. $72: adjusted costs of blood transfusion in Uganda. [Watson et al (1990)](https://www.sciencedirect.com/science/article/abs/pii/0955388690900892). This is cost of blood collection and processing.
        3. Length of stay; Median 4 days [Machini et al (2022)](https://bmjopen.bmj.com/content/12/6/e059263.long)
        """
        st.write(additional_notes)

with tabs[4]:
    with st.expander("Contact us"):
        contact_form = """
            <form action="https://formsubmit.co/obierochieng@gmail.com" method="POST">
                <input type="hidden" name="_captcha" value="false">
                <input type="text" name="name" placeholder="Your name" required>
                <input type="email" name="email" placeholder="Your email" required>
                <input type="text" name="_honey" style="display:none">
                <input type="hidden" name="_cc" value="ocu9@cdc.gov">
                <textarea name="message" placeholder="Details of your problem"></textarea>
                <button type="submit">Send Information</button>
            </form>
        """    
        st.markdown(contact_form, unsafe_allow_html=True)

        def local_css(file_name):
            with open(file_name) as f:
                st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

        local_css("style/style.css")
