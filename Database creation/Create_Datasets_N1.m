% this file creates datasets
% -----FIRST STAGE------
% Based on historical data of loads/renewables:
% compute optimal (PD,PG,VG) with the N-1
% security-constrained AC-OPF
% -----SECOND STAGE------
% Fit multi-variate normal distribution and draw additional samples
% (PD,PG,VG) these sample are classified according to feasibility with the N-1
% security-constrained AC-OPF
% author: Andreas Venzke; andven@elektro.dtu.dk
% DISCLAIMER: The code is not tested extensively; it may still have bugs/need to be modified!

%cd '/Users/Eigil/Dropbox/DTU/Speciale/matpower7.0'
%cd '/Users/Eigil/Dropbox/DTU/Speciale/Data Generation/Eigil_Dataset_N1/matpower6.0/matpower6.0'
addpath('/Users/Eigil/Dropbox/DTU/Speciale/Data Generation/Eigil_Dataset_N1/matpower6.0/matpower6.0')
%Matpower_addpath;  


close all;
clear all;
tic();

% ----- INITIALIZE ------

% This loads many of the indices for the mpc file and makes the indexing
% easier (check matpower manual for more details)
% define named indices into data matrices
[PQ, PV, REF, NONE, BUS_I, BUS_TYPE, PD, QD, GS, BS, BUS_AREA, VM, ...
    VA, BASE_KV, ZONE, VMAX, VMIN, LAM_P, LAM_Q, MU_VMAX, MU_VMIN] = idx_bus;
[GEN_BUS, PG, QG, QMAX, QMIN, VG, MBASE, GEN_STATUS, PMAX, PMIN, ...
    MU_PMAX, MU_PMIN, MU_QMAX, MU_QMIN, PC1, PC2, QC1MIN, QC1MAX, ...
    QC2MIN, QC2MAX, RAMP_AGC, RAMP_10, RAMP_30, RAMP_Q, APF] = idx_gen;
[F_BUS, T_BUS, BR_R, BR_X, BR_B, RATE_A, RATE_B, RATE_C, ...
    TAP, SHIFT, BR_STATUS, PF, QF, PT, QT, MU_SF, MU_ST, ...
    ANGMIN, ANGMAX, MU_ANGMIN, MU_ANGMAX] = idx_brch;
[PW_LINEAR, POLYNOMIAL, MODEL, STARTUP, SHUTDOWN, NCOST, COST] = idx_cost;


% download test cases at https://github.com/power-grid-lib/pglib-opf
% these PGLIB-OPF test cases are new modern benchmarks and have relevant constraint
% limits defined; the testcases in MATPOWER have sometimes limits not
% specified

% the number in the test case refers to the number of buses
% mpc = case9;% this test case if from MATPOWER
% mpc = case39; % this test case is from PGLIB-OPF
mpc = case118; % this test case is from PGLIB-OPF

% define constraint limit tolerance
% Please check in check_lims how the tolerance is defined
% Had to update this as otherwise N-1 SC-AC-OPF solution does not satisfy
% limits
tol = 0.001; % corresponds to 0.1% based on the difference of minimum and maximum constraint limit

% get number of buses
nb = size(mpc.bus,1);
% get number of lines
nl = size(mpc.branch,1);
% get number of generators (including synchronous generators)
ng = size(mpc.gen,1);

% define line outages (these are the numbers of the lines we remove later)
% Note that small test cases are usually not secure against ALL possible
% line outages
% the more outages are included the longer it will take to solve the N-1
% SC-AC-OPF
if nb == 9
    line_outages = [2;3;5;6;8;9]; % outages for IEEE 9 bus system
elseif nb == 39
    % I selected these contingencies to have feasible N-1 SC-AC-OPFs (while the AC-OPF solution did not satsify the constraints)
    line_outages = [1 2 3 4 10 11 12 13 40].';
elseif nb == 118
    line_outages = [1 5 6 10 11 16 17 18 19 20 45 46 ].';
end 

% number of line outages/contingencies
nr_con = size(line_outages,1)

% indices of voltage set-points of generators
v_IDs = mpc.gen(:,GEN_BUS);

% generator indices
ID_gen = find(mpc.gen(:,PMAX)-mpc.gen(:,PMIN)>=10^-4); %only look at generators
% It is important that some of the generators are synchronous condenser
% and their active power set-point is fixed

[maxP,id_maxP] = max(mpc.gen(:,PMAX)-mpc.gen(:,PMIN));
% adjust slack bus; to get more feasible samples make the slack bus the
% largest generator
fprintf('Attention: setting slack bus to the largest generator: G%d\n',id_maxP);
mpc.bus(mpc.bus(:,2)==REF,2) = PV;
mpc.bus(mpc.gen(id_maxP,GEN_BUS),2) = REF;

[maxP2,id_maxP2] = max(mpc.gen(ID_gen,PMAX)-mpc.gen(ID_gen,PMIN));
% remove slack bus from ID_gen
% active power of generator compensates the change in losses
ID_gen(id_maxP2) = [];

% load and create historical data for first stage
time_steps = 1000;%8760;
time_steps = 8760;

% second stage number of samples
nr_samples = 10000;


% https://github.com/LucienBobo/SOCP-OPF-Appendix/tree/master/Profiles
% PV and load data we used in another project
% TODO: Maybe there is better data available somewhere? Multiple
% years/multiple locations?
% Both profiles are obtained from datesets for Chicago Midway, 2010
% Load data from https://openei.org/datasets/files/961/pub/RESIDENTIAL_LOAD_DATA_E_PLUS_OUTPUT/BASE/
% PV data from https://rredc.nrel.gov/solar/old_data/nsrdb/1991-2010/hourly/siteonthefly.cgi?id=725340

% loading the demand/load data P_D
% data is normalized between 0 and 1
% hist_load = rand(1,time_steps); % random data 
cd '/Users/Eigil/Dropbox/DTU/Speciale/Data Generation/Eigil_Dataset_N1'
hist_load = csvread('./data_profiles/load.csv');
%hist(hist_load);
% loading the RES data
% this is PV data so at night there is zero output
hist_RES = csvread('./data_profiles/PV.csv');
%hist(hist_RES);
%hist_RES = rand(1,time_steps); % random data 


% projecting/distributing the data according to the different loads
% draw from a multivariate distribution with standard deviation of 0.1% of
% P_initial

% identify non-zero loads!
ID_loads = find(mpc.bus(:,PD)~=0);
nloads = size(ID_loads,1);
% map from buses to loads
mapload2b = zeros(nb,nloads);
for i = 1:nloads
mapload2b(ID_loads(i,1),i)=1;
end

% draw time_steps samples from Nbus-dimensional multivariate normal
% distribution
sigma = 0.1*eye(nloads,nloads); % no correlation currently; standard deviation 10%
% TODO: please feel free to change this
sample_load = zeros(time_steps,nloads);
sample_RES = zeros(time_steps,nloads); 
% the final composition of loads and RES
P_load_sample=zeros(time_steps,nloads);

% for each time step, we assume the mean is the historical value for that
% time stamp and the standard deviation is 10% of rated load
for t = 1:time_steps
sample_load(t,:) = mvnrnd(ones(nloads,1).*hist_load(t,1),sigma);

% if solar output is 0 (at night, then output at all buses is zero as well)
if hist_RES(t,1)==0.0
    sample_RES(t,:) = zeros(nloads,1);
else
   sample_RES(t,:) = mvnrnd(ones(nloads,1).*hist_RES(t,1),sigma);
end 

% we need to saturate each random sample between 0 and 1 as we do not want
% to have output outside the bounds (infeasibility issues)
sample_load(t,:) = max(sample_load(t,:) ,0);
sample_load(t,:) = min(sample_load(t,:) ,1);
sample_RES(t,:) = max(sample_RES(t,:) ,0);
sample_RES(t,:) = min(sample_RES(t,:) ,1);


% Current idea
% P_load = 75%*P_load_initial - %_RES * 25%*P_load_initial + %_LOAD* 25%*P_load_initial
% Essentially 75% is fixed load; 25% is load fluctuation and 25% is
% renewable injection
% This ensures that overall the variation is between 50% and 100% (otherwise
% there might be feasbility issues)
P_load_sample(t,:) = 0.75*mpc.bus(ID_loads,PD)+(sample_load(t,:).').*abs(0.25*mpc.bus(ID_loads,PD)) - (sample_RES(t,:).').*abs(0.25*mpc.bus(ID_loads,PD));

end 

% to clarify, we will use the P_load_sample data in the first stage to
% compute the optimal set-points

% look at the structure of the data
% for e.g. for 9 bus system
% hist(P_load_sample(:,5)); % hist(P_load_sample(:,7)); % hist(P_load_sample(:,9));


% ----- FIRST STAGE ------
% number of samples in first stage: time_steps
% this is for the N-1 SC-AC-OPF
Viol_log = zeros(time_steps,nr_con+1,4); % Constraint Types Violated: PG QG Vbus Sline
Viol_mag_log = zeros(time_steps,nr_con+1,4); % Constraint Types Magnitudes: PG QG Vbus Sline
success_log = zeros(time_steps,nr_con+1); % AC power flow success

% this is for the AC-OPF
Viol_log_onlyAC = zeros(time_steps,nr_con+1,4); % Constraint Types Violated: PG QG Vbus Sline
Viol_mag_log_onlyAC = zeros(time_steps,nr_con+1,4); % Constraint Types Magnitudes: PG QG Vbus Sline
success_log_onlyAC = zeros(time_steps,nr_con+1); % AC power flow success

cost_of_security_log = zeros(time_steps,1);

% dataset includes
% PD at load buses; nr_loads
% PG at all generators except synchronous condensers and slack bus
% VG at all generators ng
ngen_wo_slack_syn = size(ID_gen,1);
disp(ngen_wo_slack_syn)
% dataset is normalized
% maybe this container can be programmed more efficiently
dataset_first_stage = zeros(nloads+ngen_wo_slack_syn+ng,time_steps);
dataset_max = [max(mpc.bus(ID_loads,PD),0.5*mpc.bus(ID_loads,PD)); mpc.gen(ID_gen,PMAX); mpc.bus(mpc.gen(:,GEN_BUS),VMAX)];
dataset_min = [min(mpc.bus(ID_loads,PD),0.5*mpc.bus(ID_loads,PD)); mpc.gen(ID_gen,PMIN); mpc.bus(mpc.gen(:,GEN_BUS),VMIN)];

% keep unmodified struct
mpc_original = mpc;
% power factor of loads; is kept constant
power_factor = mpc.bus(ID_loads,QD)./mpc.bus(ID_loads,PD);

tStart_first = tic;  
% for debugging use for loop instead of parfor
parfor t = 1:time_steps % index of first stage
    disp(t)
    % adjust the loading of the case according to historical data
    % TODO: adjust this based on chosen method
    mpc = mpc_original;
    % We adjust the reactive power load according to constant power factor
    % QD_new = PD_new * (QD_initial/PD_initial)
    mpc.bus(:,QD) = mapload2b*((P_load_sample(t,:).').*power_factor);
    % adjust active power load
    mpc.bus(:,PD) = mapload2b*(P_load_sample(t,:).');
        

    % this builds the N-1 security constrained AC-OPF
    % line_outages: number of branches that are outaged
    % ID_gen: generators for which the active power set-points are forced to be
    % equal (Remember NOT to include slack bus, as losses change and have to be
    % compensated)
    % v_IDs: generators for which the generator voltage set-points are forced
    % to be equal
    mpc_N1 = build_N1(mpc,line_outages,ID_gen,v_IDs);
    
    
    % IPOPT is faster and more stable than MATPOWER solver
    opt_opf = mpoption; 
%    opt_opf.opf.ac.solver='IPOPT';
    % this surpresses most of the output of the AC optimal power flow command
    opt_opf.verbose = 0;
    opt_opf.out.all = 0;
    
    % solve AC-OPF
    results = runopf(mpc,opt_opf);
    
    % solve preventive N-1 security constrained AC-OPF
    results_N1=runopf(mpc_N1,opt_opf);
    
    
    IDs0 = (nloads+1):(nloads+ngen_wo_slack_syn);
    %dataset_first_stage(IDs0,t)=(P_load_sample(t,:).')./(dataset_max(1:nloads,1)-dataset_min(1:nloads,1));
    % extract dataset set-points
    IDs1 = (nloads+1):(nloads+ngen_wo_slack_syn);
    %dataset_first_stage(IDs1,t)=(results_N1.gen(ID_gen,PG)-dataset_min(IDs1,1))./(dataset_max(IDs1,1)-dataset_min(IDs1,1));
    IDs2 = (nloads+ngen_wo_slack_syn+1):(nloads+ngen_wo_slack_syn+ng);
    %dataset_first_stage(IDs2,t)=(results_N1.gen(1:ng,VG)-dataset_min(IDs2,1))./(dataset_max(IDs2,1)-dataset_min(IDs2,1));
    
    % to have parfor (parallel matlab loop) working we need
    dataset_first_stage(:,t) = [(P_load_sample(t,:).'-dataset_min(1:nloads,1))./(dataset_max(1:nloads,1)-dataset_min(1:nloads,1));...
        (results_N1.gen(ID_gen,PG)-dataset_min(IDs1,1))./(dataset_max(IDs1,1)-dataset_min(IDs1,1));...
        (results_N1.gen(1:ng,VG)-dataset_min(IDs2,1))./(dataset_max(IDs2,1)-dataset_min(IDs2,1))];
    
    if results_N1.success == 0 || results.success == 0
        error('OPF did not converge -- abort -- Recommendation: change contingencies')
    end
    
    % cost increase considering N-1 security (in percent)
    % Should always be larger or equal to 0.0
    cost_of_security_log(t) = ((results_N1.f/results.f)-1)*100;
    
    % loop over possible contingencies
    % This ONLY serves to validate that the obtained N-1 SC-AC-OPF does
    % actually satisfy the constraints
    Viol_ = [];
    Viol_mag_ = [];
    success_ = [];
    for con = 1:nr_con+1
        mpc_out = mpc;
        % setpoints from N-1 security constrained AC-OPF
        mpc_out.gen(:,PG) = results_N1.gen(1:ng,PG);
        mpc_out.gen(:,VG) = results_N1.gen(1:ng,VG);
        if con == 1
            % do nothing -- this is the intact system state (no
            % outages)
        else
            % remove line corresponding to index in line_outages
            % from the mpc file
            mpc_out.branch(line_outages(con-1,1),:)=[];
            
            % check if branch outage splits network (this should not
            % happen)
            [a,isolated_buses] = find_islands(mpc_out);
            if ~isempty(isolated_buses)
                error('Isolated buses are created -- choose other contingencies');
            end
        end
        % evaluate constraint violations
        [Viol,Viol_mag,success] = check_lims(mpc_out,tol);
        Viol_ = [Viol_; Viol];
        Viol_mag_ = [Viol_mag_; Viol_mag];
        success_ = [success_; success];
    end
    
    % this is necessary for parallel loop
    Viol_log(t,:,:) = Viol_;
    Viol_mag_log(t,:,:) = Viol_mag_;
    success_log(t,:) = success_;
    
    % loop over possible contingencies
    % This serves to check whether the AC OPF (without contingencies) would
    % satisfy the constraints; this is an additional check; can be
    % deactivated/removed for comp. efficiency
        Viol_ = [];
    Viol_mag_ = [];
    success_ = [];
    for con = 1:nr_con+1
        mpc_out = results;
        if con == 1
            % do nothing -- this is the intact system state (no
            % outages)
        else
            % remove line corresponding to index in line_outages
            % from the mpc file
            mpc_out.branch(line_outages(con-1,1),:)=[];
            
            % check if branch outage splits network (this should not
            % happen)
            [a,isolated_buses] = find_islands(mpc_out);
            if ~isempty(isolated_buses)
                error('Isolated buses are created -- choose other contingencies');
            end
        end
        
        % evaluate constraint violations
        [Viol,Viol_mag,success] = check_lims(mpc_out,tol);
               Viol_ = [Viol_; Viol];
        Viol_mag_ = [Viol_mag_; Viol_mag];
        success_ = [success_; success];

    end
            Viol_log_onlyAC(t,:,:) = Viol_;
        Viol_mag_log_onlyAC(t,:,:) = Viol_mag_;
        success_log_onlyAC(t,:) = success_;
  
end

% evaluate convergence of AC power flows
share_conv_onlyAC=sum(sum(success_log_onlyAC))/((nr_con+1)*time_steps);
share_conv=sum(sum(success_log))/((nr_con+1)*time_steps);

if share_conv ~= 1 || share_conv_onlyAC ~=1
    error('Not all AC PFs did converge')
end

% evaluate if any of the N-1 SC-AC-OPF solution does violate constraints
% if yes, throw error
share_feas = sum(sum(sum(Viol_log,2),3)==0)/time_steps;
share_feas_onlyAC = sum(sum(sum(Viol_log_onlyAC,2),3)==0)/time_steps;

% largest constraint violation
max_infeas = max(max(Viol_mag_log,[],2),[],1);
max_infeas_onlyAC = max(max(Viol_mag_log_onlyAC,[],2),[],1);
% time meas
tElapsed_first = toc(tStart_first);

% print summary
fprintf(' ------------- Completed first stage in %4.0f seconds ---------------------- \n',tElapsed_first);
fprintf('Number of samples:                  %6.0f \n',time_steps);
fprintf(' -------------------------------------------------------------------------- \n');
fprintf('Share of feasible samples:          %6.2f %% (AC-OPF)  %6.2f %% (N-1 SC-ACOPF) \n',share_feas_onlyAC*100,share_feas*100);
fprintf('Share of converged AC power flows:  %6.2f %% (AC-OPF)  %6.2f %% (N-1 SC-ACOPF) \n',share_conv_onlyAC*100,share_conv*100);
fprintf(' -------------------------------------------------------------------------- \n');
fprintf('Largest violations (AC-OPF):        %6.2f (PG) %6.2f (QG) %6.2f (Vbus) %6.2f (Sline) \n', max_infeas_onlyAC(:,:,1),max_infeas_onlyAC(:,:,2),max_infeas_onlyAC(:,:,3),max_infeas_onlyAC(:,:,4)) 
fprintf('Largest violations (N-1 SC-AC-OPF): %6.2f (PG) %6.2f (QG) %6.2f (Vbus) %6.2f (Sline) \n', max_infeas(:,:,1),max_infeas(:,:,2),max_infeas(:,:,3),max_infeas(:,:,4)) 
fprintf(' -------------------------------------------------------------------------- \n');


%save('dataset_first_stage.mat', 'dataset_first_stage')

% ----- SECOND STAGE ------

% fit a multivariate normal distribution (MVND) to sample database (PD,PG,VG)
% TODO: maybe there are better distributions here? How to choose the
% covariance matrix, e.g. reduce/enlarge?
% Gaussian Mixture Model?

% Maximum Likelihood Estimation yields

Mskekur(dataset_first_stage', 1, 0.05) %Extremely non-Gaussian

%Looking at the marginal distributions
hold on
for i=1:size(dataset_first_stage, 1)
    ksdensity(dataset_first_stage(i, :))
end
xlim([-0.2, 1.2])
ylim([0, 2])



%% Approach 1: One Gaussian Mixture

% cov_data = cov(dataset_first_stage.');
% cov_data = (time_steps-1)/time_steps*cov_data; %MLE estimation
% mean_data = mean(dataset_first_stage.').';
% % % amount of samples drawn from MVND
% % nr_samples = 100000;
% 
% % draw samples
% % red_cov = 0.5; % to reduce covariance
% dataset_second_stage = mvnrnd(mean_data,cov_data,nr_samples).';
%% Approach 2: Projected GMM
% Second approach: Fitting a GMM
GMM = fitgmdist(dataset_first_stage', 9, 'RegularizationValue', 1e-6);
nr_samples = 1e5;
dataset_second_stage = random(GMM, nr_samples)';

%See how many samples that have values below 0
mean(any(dataset_second_stage < 0, 1)) %99.34 of samples have a feature below 0
mean(mean(dataset_second_stage < 0)) %2.83% of sampled features are below 0

%See how many samples that have values above 1
mean(any(dataset_second_stage > 1, 1)) %99.98 of samples have a feature below 1
mean(mean(dataset_second_stage > 1)) %5.44% of sampled features are below 1%

%
% saturate based on limits (PG, VG)
% it does not make sense to analyse inputs outside control
% variable bounds (as these are directly infeasible)
% saturate data between 0 and 1
%dataset_second_stage=max(dataset_second_stage,0);
%dataset_second_stage=min(dataset_second_stage,1);


%% Approach 2: GMM with rejection sampling
%Rejection sampling and mixing with 5000 observations outside the domain
GMM = fitgmdist(dataset_first_stage', 9, 'RegularizationValue', 1e-12);
nr_samples = 1e5;


pi = GMM.ComponentProportion;
component_draw = mnrnd(nr_samples, pi);
[n_variables, ~] = size(dataset_first_stage);
ul = ones(n_variables, 1);
ll = zeros(n_variables, 1);

cd '/Users/Eigil/Dropbox/DTU/Speciale/Data Generation/Truncated_Gaussian'

dataset_second_stage = [];
for i = 1:length(pi)
    nr_draws = component_draw(i);
    mu = GMM.mu(i, :)';
    Sigma = GMM.Sigma(:, :, i);
    %samples = rmvnrnd(mu, Sigma, nr_draws, [-eye(n_variables); eye(n_variables)], [-ll; ul]);
    samples = mu + mvrandn(ll - mu, ul - mu, Sigma, nr_draws);
    dataset_second_stage = [dataset_second_stage, samples];
end

%Mixing with 5000 non-truncated samples
dataset_second_stageB = random(GMM, 5000)';

dataset_second_stage = [dataset_second_stage, dataset_second_stageB];
idx = randperm(length(dataset_second_stage));
dataset_second_stage = dataset_second_stage(:, idx);

nr_samples = size(dataset_second_stage, 2);


%% Latin Hypercube sampling
%nr_samples = 1e5;
%dataset_second_stage = lhsdesign(nr_samples, size(dataset_first_stage, 1));
%dataset_second_stage = dataset_second_stage';


%%
Viol_log2 = zeros(nr_samples,nr_con+1,4); % Constraint Types Violated: PG QG Vbus Sline
Viol_mag_log2 = zeros(nr_samples,nr_con+1,4); % Constraint Types Magnitudes: PG QG Vbus Sline
success_log2 = zeros(nr_samples,nr_con+1); % AC power flow success

tStart_second = tic;  
% loop over samples and classifiy according to feasibility
for i = 1:nr_samples
    if mod(i, 100) == 0
        disp(i)
    end
    % set the corresponding set-points
    mpc = mpc_original;
    
    % active load
    mpc.bus(:,PD) = mapload2b*(dataset_min(1:nloads,1)+dataset_second_stage(1:nloads,i).*(dataset_max(1:nloads,1)-dataset_min(1:nloads,1)));
    % reactive load
    mpc.bus(ID_loads,QD) = mpc.bus(ID_loads,PD).*mpc_original.bus(ID_loads,QD)./mpc_original.bus(ID_loads,PD);%mapload2b*(dataset_min(1:nloads,1)+dataset_second_stage(1:nloads,i).*(dataset_max(1:nloads,1)-dataset_min(1:nloads,1)).*power_factor);
    % active generator set-points
    IDs1 = (nloads+1):(nloads+ngen_wo_slack_syn);
    mpc.gen(ID_gen,PG) = dataset_min(IDs1,1)+(dataset_second_stage(IDs1,i)).*(dataset_max(IDs1,1)-dataset_min(IDs1,1));
    % voltage generator set-points
    IDs2 = (nloads+ngen_wo_slack_syn+1):(nloads+ngen_wo_slack_syn+ng);
    mpc.gen(:,VG) = dataset_min(IDs2,1)+(dataset_second_stage(IDs2,i)).*(dataset_max(IDs2,1)-dataset_min(IDs2,1));    
    
    % check feasiblity 
    
    % loop over possible contingencies
    Viol_ = [];
    Viol_mag_ = [];
    success_ = [];
    for con = 1:nr_con+1
        mpc_out = mpc;
        if con == 1
            % do nothing -- this is the intact system state (no
            % outages)
        else
            % remove line corresponding to index in line_outages
            % from the mpc file
            mpc_out.branch(line_outages(con-1,1),:)=[];
            
            % check if branch outage splits network (this should not
            % happen)
            [a,isolated_buses] = find_islands(mpc_out);
            if ~isempty(isolated_buses)
                error('Isolated buses are created -- choose other contingencies');
            end
        end
        
        % evaluate constraint violations
        [Viol,Viol_mag,success] = check_lims(mpc_out,tol);
        Viol_ = [Viol_; Viol];
        Viol_mag_ = [Viol_mag_; Viol_mag];
        success_ = [success_; success];
    end
        Viol_log2(i,:,:) = Viol_;
        Viol_mag_log2(i,:,:) = Viol_mag_;
        success_log2(i,:) = success_;
end 

% evaluate convergence of AC power flows
share_conv2=sum(sum(success_log2))/((nr_con+1)*nr_samples);

% evaluate constraint violations
share_feas2 = sum(sum(sum(Viol_log2,2),3)==0)/nr_samples;

% largest constraint violation
max_infeas2 = max(max(Viol_mag_log2,[],2),[],1);

tElapsed_second = toc(tStart_second);
  
fprintf(' ------------- Completed second stage in %4.0f seconds ---------------------- \n',tElapsed_second);
fprintf('Number of samples:                  %6.0f \n',nr_samples);
fprintf(' -------------------------------------------------------------------------- \n');
fprintf('Share of feasible samples:          %6.2f %% \n',share_feas2*100);
fprintf('Share of converged AC power flows:  %6.2f %%  \n',share_conv2*100);
fprintf(' -------------------------------------------------------------------------- \n');
fprintf('Largest violations:        %6.2f (PG) %6.2f (QG) %6.2f (Vbus) %6.2f (Sline) \n', max_infeas2(:,:,1),max_infeas2(:,:,2),max_infeas2(:,:,3),max_infeas2(:,:,4)) 
fprintf(' -------------------------------------------------------------------------- \n');


% assemble the data for neural network training
% not clear how to treat samples in the first stage that due to some
% numerical inaccuracies are not classified as feasible
% here we include them and classify them as infeasible
classification_first = sum(sum(Viol_log,2),3)==0;
classification_second = sum(sum(Viol_log2,2),3)==0;

% remove all data from the second stage where 
% AC power flows did not converge
IDs_converged_first = find(sum(success_log,2)==(nr_con+1));
IDs_converged_second = find(sum(success_log2,2)==(nr_con+1));
dataset_first_stage_red = dataset_first_stage(:,IDs_converged_first);
dataset_second_stage_red = dataset_second_stage(:,IDs_converged_second);

classification_first_red = classification_first(IDs_converged_first);
classification_second_red = classification_second(IDs_converged_second);

data_input = [dataset_first_stage_red dataset_second_stage_red];
data_output = [classification_first_red;classification_second_red];


data_size = size(data_input,2);
  
fprintf(' ------------- Summary ---------------------- \n');
fprintf('Number of samples:                  %6.0f \n',data_size);
fprintf('Number of feasible samples:         %6.0f (%6.2f %%) \n',sum(data_output),sum(data_output)./data_size * 100);
fprintf('Number of infeasible samples:       %6.0f (%6.2f %%) \n',(data_size-sum(data_output)),(data_size-sum(data_output))./data_size*100);

% saving data for import to Tensorflow/Pytorch
save(strcat('data_input_',num2str(nb)),'data_input');
save(strcat('data_output_',num2str(nb)),'data_output');
save(strcat('dataset_min_',num2str(nb)),'dataset_min');
save(strcat('dataset_max_',num2str(nb)),'dataset_max');
% save entire workspace
save(strcat('workspace_',num2str(nb)));


X = data_input';

y = data_output;
X_min = dataset_min';
X_max = dataset_max';
%save('Big1.mat', 'X', 'y', 'X_min', 'X_max') %Big 1: GMM (8 components,
%non-truncated)

save('Big2.mat', 'X', 'y', 'X_min', 'X_max') %Big 2: GMM (9 components
%100,000 truncated observations, 5,000 non-truncated observations


