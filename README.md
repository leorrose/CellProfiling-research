# CellProfiling-research

##Instructions
### Preperations
```bash
conda create --name tf-env python=3.8.6 tensorflow-gpu=2.2.0 scikit-learn pandas tqdm progressbar2 matplotlib
```
> To create an enviroment named tf-env, need to be executed once.

If we want to be able to run a jupyter notebook server we will need to run `conda install --name tf-env jupyterlab`

------------
### Execute a batch job
There are some batch files currently in our work folder
1. `sbatch-downloader` for downloading plates
2. `sbatch-run-diagnose-all` for running the main project for the big plates folder
3. `sbatch-run-diagnose-few`for running the main project for the smaller folder of plates named "few"
4. `sbatch-notebook` for running a jupyter notebook server

> In the sbatch files we can redefine various things like scripts' parameter, email notifications, etc.

In order to execute a job we need to do the following thing
`sbatch <sbatch-file-name>`

Extras:
- We can define a dependacy between excuting jobs like
`sbatch --dependency=afterok:<other_job_id> <sbatch-file-name>`
a possible usage for that is running with the big plate folder only after it was successfuly executed with the small folder

- At our lab we have a few "golden tickets" which will give us priority for nodes. We can use it wisely if we want to ran with less chance to be preempted.
`sbatch --qos=assafzar <sbatch-file-name>`

------------

### Scripts
#### /code/learning/main.py
##### Usages
- Usage 1: without parameters, will run over the default big directory
- Usage 2: `main.py <working_folder>` will run over the given directory

Note: Currently the directory must contain the "csvs" folder from the downloader script

##### Output
Generates output in the given directory in a folder named "results"

#### /code/plateDownloader.py
##### Usages
- Usage 1: `plateDownloader.py <working_folder> -n <plate_amount>`
- Usage 2: `plateDownloader.py <working_folder> -l <plate_number1> <plate_number2> ...`
- Usage 3: `plateDownloader.py <working_folder> -n <plate_amount> -l <plate_number1> <plate_number2> ...`

Notes:
- Will download files that not already exists in the given folders
- With the "n" parameter, selects n plates randomly from the given list or from the ftp server


##### Output
Generates three folders in the given directory
- tars: contain the raw tar.gz files of the plates
- extracted: the relevant files from the compressed files
- csvs: the relevant details for each plate in a csv file per plate, contain all the details for the cells from this plate.
