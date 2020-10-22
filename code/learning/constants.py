# # Constants and params


# In[1]:


# change directory to project directory within the department cluster (SLURM)
# PROJECT_DIRECTORY = r'/storage/users/g-and-n/plates/'
PROJECT_DIRECTORY = r'C:\Users\Niko\Desktop\plates'

CHANNELS = ["AGP", "DNA", "ER", "Mito", "RNA"]
LABEL_FIELD = 'Metadata_ASSAY_WELL_ROLE'
S_STD = 'Std'
S_MinMax = 'MinMax'
ERROR_TYPE = 'RMSE'  # 'RMSE' | 'MSE' | 'MAE'
