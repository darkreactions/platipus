from models.meta.platipus_class import Platipus
from hpc_scripts.hpc_params import (common_params, local_meta_params,
                                    local_meta_train)
# from models.meta.init_params import init_params
from utils import (initialise_dict_of_dict, save_model,
                   update_cv_stats_dict, write_pickle)
import pickle
import shap
import matplotlib.pyplot as plt

feature_name = ["_rxn_M_acid", "_rxn_M_inorganic", "_rxn_M_organic", "_solv_GBL", "_solv_DMSO", "_solv_DMF", "_stoich_mmol_org",	"_stoich_mmol_inorg", "_stoich_mmol_acid", "_stoich_mmol_solv",	"_stoich_org-solv",	"_stoich_inorg-solv", "_stoich_acid-solv", "_stoich_org+inorg-solv", "_stoich_org+inorg+acid-solv", "_stoich_org-liq", "_stoich_inorg-liq", "_stoich_org+inorg-liq", "_stoich_org-inorg", "_stoich_acid-inorg", "_rxn_Temperature_C",	"_rxn_Reactiontime_s", "_feat_AvgPol", "_feat_Refractivity", "_feat_MaximalProjectionArea",	"_feat_MaximalProjectionRadius",
                "_feat_maximalprojectionsize", "_feat_MinimalProjectionArea",	"_feat_MinimalProjectionRadius", "_feat_minimalprojectionsize",	"_feat_MolPol",	"_feat_VanderWaalsSurfaceArea",	"_feat_ASA", "_feat_ASA_H", "_feat_ASA_P",	"_feat_ASA-",	"_feat_ASA+", "_feat_ProtPolarSurfaceArea", "_feat_Hacceptorcount", "_feat_Hdonorcount", "_feat_RotatableBondCount",	"_raw_standard_molweight", "_feat_AtomCount_N", "_feat_BondCount",	"_feat_ChainAtomCount",	"_feat_RingAtomCount", "_feat_primaryAmine",	"_feat_secondaryAmine",	"_rxn_plateEdgeQ", "_feat_maxproj_per_N", "_raw_RelativeHumidity"]
class_name = ["failure", "success"]

params = {**common_params, **local_meta_params}
train_params = {**params, **local_meta_train}
train_params['gpu_id'] = 0
amine = 'ZEVRFFCPALTVDN-UHFFFAOYSA-N'

platipus = Platipus(train_params, amine=amine, training=False,
                    model_name=train_params['model_name'])
platipus.load_model(
    './results/Platipus_local_10_shot/ZEVRFFCPALTVDN-UHFFFAOYSA-N/drp_chem_10shot_2.pt')

val_data = pickle.load(
    open('./results/Platipus_local_10_shot/testing/val_dump.pkl', 'rb'))
x_t, y_t = val_data[amine][2], val_data[amine][3]

# print(platipus.predict_proba(x_t))

explainer = shap.KernelExplainer(platipus.predict_proba, x_t)
shap_values = explainer.shap_values(x_t, nsamples=100)


fig = shap.summary_plot(shap_values, features=x_t, feature_names=feature_name, max_display=10, class_names=[
                        "Failure", "Success"], title="Feature Importance", show=False)
#fig = shap.force_plot(explainer.expected_value[0], shap_single_values[0], x_true[i:i+1], show = False, feature_names = feature_name, matplotlib = True, text_rotation = 10, figsize = (20, 5))

fig_name = "Platipus_shap_" + amine + "_" + ".png"
plt.savefig(fig_name, bbox_inches='tight')
plt.close()
