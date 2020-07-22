% for a power system test case create a dataset with
% features: P_G (TODO: consider generator voltages V_G and loading P_D, Q_D as well)
% classfication: secure (1) / not secure (0) w.r.t N-1 security criterion
clear all;
close all;
rng(1) % Seed. 1 = Validation set

%Install matpower
cd '/Users/Eigil/Dropbox/DTU/Speciale/matpower7.0'
Matpower_addpath;

cd '/Users/Eigil/Dropbox/DTU/Speciale/Data Generation'


% download test cases at https://github.com/power-grid-lib/pglib-opf
% these PGLIB-OPF test cases are new modern benchmarks and have relevant constraint
% limits defined; the testcases in MATPOWER have sometimes limits not
% specified

% the number in the test case refers to the number of buses
mpc = case9;

%Turn on/off sampling of different variables
VG_sampling=true;
PD_sampling=true;


% check
% https://www.researchgate.net/figure/Sample-test-system-the-IEEE-9-bus-grid_fig1_308202650
% to see how 9 bus grid looks like

% good small test cases to use are
% IEEE 9 bus: case9 (MATPOWER test case)
% IEEE 14 bus: pglib_opf_case14_ieee  
% IEEE 30 bus: pglib_opf_case30_ieee
% you can also define your own test cases using the mpc format (e.g. a
% simple 3 bus test case)


% define line outages (these are the numbers of the lines we remove later)
% Note that small test cases are usually not secure against ALL possible
% line outages (EIGIL: Changed to column vector)
line_outages = [2;3];
% number of line outages/contingencies
nr_con = size(line_outages,1);

% familiarize yourself with contents of mpc.bus, mpc.gen, mpc.branch

% This loads many of the indices for the mpc file and makes the indexing
% easier (check matpower manual for more details
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


% get number of buses
nb = size(mpc.bus,1);
% get number of lines
nl = size(mpc.branch,1);
% get number of generators (including synchronous generators)
ng = size(mpc.gen,1);


% Note: some of the larger test cases have multiple generators connected to
% one bus, or multiple parallel lines (i.e. connecting the same bus)
% TODO: this code does not take this into account
if size(unique(mpc.gen(:,GEN_BUS)),1) ~= ng
    error('multiple generators at one bus -- this code does not support this currently')
end

if size(unique(mpc.branch(:,[F_BUS T_BUS]),'rows'),1) ~= nl
    error('multiple parallel branches -- this code does not support this currently')
end



% set voltage set-point to OPF solution (this can be later varied as well)

% define options of power flow
opf_opt = mpoption;

% this surpresses most of the output of the power flow command
opf_opt.verbose = 0;
opf_opt.out.all = 0;

%Run optimal flow to set Vg (voltage magnitude set-point) 
results_opf = runopf(mpc,opf_opt);
mpc.gen(:,VG) = results_opf.gen(:,VG);
if results_opf.success ~= 1
     error('OPF did not converge -- issues with test case. Abort');
end

%Eigil: Relaxation of all voltage constraints with 0.5%
epsilon = 0.05;


% get the number of synchronous generators (these are generators that ONLY
% provide reactive power, i.e. active power P_G = 0; P_G_min = 0; P_G_max =
% 0)
ID_synch = find(mpc.gen(:,PMAX) == mpc.gen(:,PMIN));

% those are the ID of the generators we are interested in
ID_gen = find(mpc.gen(:,PMAX) ~= mpc.gen(:,PMIN));

% we exclude the slack bus generator as this one has to compensate the
% losses and is not a degree of freedom
% the slack bus is also sometimes called reference bus
ID_slack = find(mpc.bus(:,BUS_TYPE) == REF);
% find the generator that is connected to the bus
ID_slack_gen = find(mpc.gen(ID_gen,GEN_BUS) ==ID_slack);

% eliminate slack bus from the IDs
ID_gen_red = ID_gen;
ID_gen_red(ID_slack_gen) =[];


%find demand busses
ID_demand = find(mpc.bus(:, PD) ~= 0);
ID_demand_Q = find(mpc.bus(:, QD) ~= 0);

%If there's no correspondence between active and reactive demand
if ~all(ID_demand_Q == ID_demand)
    error('Active and reactive demands doesnt match')
end


% define size of feature space
nr_feat_gen = size(ID_gen_red,1);

% define number of samples
nr_samples = 5* 10^4;
%nr_samples = 10;


% we draw samples in the feature space
% using Latin Hypercube Sampling
% this covers the input space evenly
% this is what we have been using for larger input dimensionality

% TODO: for bayesian NN we might want to bias input sampling
% note that the input features are already normalized between
% 0 - minimum generator dispatch PG_min and minimum generator voltage
% 1 - maximum generator dispatch PG_max

%input_lhs_PG = lhsdesign(nr_samples,nr_feat_gen);
%input_lhs_VG = lhsdesign(nr_samples, nr_feat_gen);


input_lhs = lhsdesign(nr_samples, 2*nr_feat_gen+1); %PG and VG
sigma = 0.25/2.5; %98.7% of points are within +/- 0.25 (2.5 std)
input_PD = normrnd(1, 0.25/3, [nr_samples, length(ID_demand)]);




% output classification
output = zeros(nr_samples,1);
% keeps the classification for intact and each individual outaged system
% state
output_outages = zeros(nr_samples,nr_con+1);

% active power line flows --> these could be used for regression (and we
% calculate them anyway)
% note that for outaged lines the flow is set to zero
% note that we have different line flows in two directions FROM / TO a bus
% due to both active and reactive power losses along the line
active_lf_to = zeros(nr_samples,nl,nr_con+1);
active_lf_from = zeros(nr_samples,nl,nr_con+1);

% apparent power line flows --> these could be used for regression (and we
% calculate them anyway)
apparent_lf_to = zeros(nr_samples,nl,nr_con+1);
apparent_lf_from = zeros(nr_samples,nl,nr_con+1);

% keep track of instances for which the power flow converged
% if a lot of power flows to not converge then usually there is some
% problem with the test case
successful_pf = zeros(nr_samples,nr_con+1);

% in parallel run power flows for different set-points of generator
% TODO: parallelize loop with parfor

constraint_matrix = logical(zeros(nr_samples, 4, nr_con+1));
%severity_matrix = zeros(nr_samples, 2, nb, nr_con+1);
%severity_matrix = zeros(nr_samples, nb, 2);

%Define indices for slicing severity matrix
idx_constr_PG = 1:ng;
idx_constr_QG = idx_constr_PG(end) + (1:ng);
idx_constr_VM = idx_constr_QG(end) + (1:nb);
idx_constr_Sline = idx_constr_VM(end) + (1:nl);

%Initialising severity matrices
severity_matrix_min = nan(nr_samples, idx_constr_Sline(end), nr_con+1);
severity_matrix_max = severity_matrix_min;


for n = 1:nr_samples
    mpc_copy = mpc;
    
    % set the active generator setpoints according to the input sample
%    mpc_copy.gen(ID_gen_red,PG) = mpc_copy.gen(ID_gen_red,PMIN)+(input_lhs_PG(n,:).').*(mpc_copy.gen(ID_gen_red,PMAX)-mpc_copy.gen(ID_gen_red,PMIN));
%    mpc_copy.gen(ID_gen_red, VG) = mpc_copy.gen(ID_gen_red,VMIN)+(input_lhs_VG(n, :).').*(mpc_copy.gen(ID_gen_red,VMAX)-mpc_copy.gen(ID_gen_red, VMIN));
    mpc_copy.gen(ID_gen_red,PG) = mpc_copy.gen(ID_gen_red,PMIN)+(input_lhs(n,1:nr_feat_gen).').*(mpc_copy.gen(ID_gen_red,PMAX)-mpc_copy.gen(ID_gen_red,PMIN));
    
    %If VG sampling is turned on
    if VG_sampling
        mpc_copy.gen(ID_gen, VG) = mpc_copy.bus(ID_gen,VMIN)/(1-epsilon)+(input_lhs(n, (nr_feat_gen+1):end).').*(mpc_copy.bus(ID_gen,VMAX)/(1+epsilon)-mpc_copy.bus(ID_gen, VMIN)/(1-epsilon));    
    end
    
    if PD_sampling
        mpc_copy.bus(ID_demand, PD) = mpc_copy.bus(ID_demand, PD).*input_PD(n, :)'; 
        mpc_copy.bus(ID_demand_Q, QD) = mpc_copy.bus(ID_demand_Q, QD).*input_PD(n, :)';
    end
    %    mpc_copy.gen(ID_gen, VG) = 1.0 + rand(3, 1)*0.1;
%    disp(mpc_copy.gen(ID_gen, VG))


    % TODO: if you want to adjust the generator voltage set-points
    % mpc.gen(ID_gen,VG) = ?
    % loading levels: mpc.bus(ID_loads,PD) = ? | mpc.bus(ID_loads,QD) = ?
    % where ID_loads are the location of loads that are input features
    
    % loop over possible contingencies
    output_outages_local = zeros(nr_con,1);
    for con = 1:nr_con+1
        mpc_out = mpc_copy;
        line_ids = [1:nl].';
        
        if con == 1
            % do nothing -- this is the intact system state (no
            % outages)
        else
            % remove line corresponding to index in line_outages
            % from the mpc file
            mpc_out.branch(line_outages(con-1,1),:)=[];
            line_ids(line_outages(con-1,1),:)=[];
            
            % check if branch outage splits network (this should not
            % happen)
            
            [a,isolated_buses] = find_islands(mpc_out);
            if ~isempty(isolated_buses)
                error('Isolated buses are created -- choose other contingencies');
            end
            
        end
        % define options of power flow
        pf_opt = mpoption;
        % set to enforce reactive power limits (this option enforces
        % the reactive power limits of generators, we can discuss
        % this in more detail); reactive power limits of generators
        % are unfortunately often causing numerical problems
        pf_opt.pf.enforce_q_lims = 1;
        
        % this surpresses most of the output of the power flow command
        pf_opt.verbose = 0;
        pf_opt.out.all = 0;
        
        %Eigil: Stricter convergence
        pf_opt.pf.tol = 10^(-14);
        pf_opt.pf.max_it = 100;
        
        % run power flow
        results_pf = runpf(mpc_out,pf_opt);
        
        if results_pf.success == 1
            successful_pf (n,con)=1;
            
            % write active power line flows
            active_lf_to(n,line_ids,con) = results_pf.branch(:,PT);
            active_lf_from(n,line_ids,con) = results_pf.branch(:,PF);
            
            % write apparent power line flows
            %CHANGE Eigil: PF and QF changed to PT and QT in the to-flows
            apparent_lf_to(n,line_ids,con) = (results_pf.branch(:,PT).^2+results_pf.branch(:,QT).^2).^(0.5);
            apparent_lf_from(n,line_ids,con) = (results_pf.branch(:,PF).^2+results_pf.branch(:,QF).^2).^(0.5);
            
            
            % check satisfaction of constraints
            % active power generator limits
            Constr_PG = any(mpc_out.gen(:,PMAX)<results_pf.gen(:,PG))+any(mpc_out.gen(:,PMIN)>results_pf.gen(:,PG));
            % reactive power generator limits
            Constr_QG = any(mpc_out.gen(:,QMAX)<results_pf.gen(:,QG))+any(mpc_out.gen(:,QMIN)>results_pf.gen(:,QG));
            % voltage magnitudes limits
            Constr_VG = any(mpc_out.bus(:,VMAX)*(1+epsilon)<results_pf.bus(:,VM))+any(mpc_out.bus(:,VMIN)*(1-epsilon)>results_pf.bus(:,VM));
            % we consider apparent branch flow limits
            % compute apparent branch flows in both directions
            % S^2 = P^2 + Q^2 holds true as S is comples phasor: S = P +
            % jQ
            Sline_to = (results_pf.branch(:,PF).^2+results_pf.branch(:,QF).^2).^(0.5);
            Sline_from = (results_pf.branch(:,PT).^2+results_pf.branch(:,QT).^2).^(0.5);
            Constr_Sline = any(mpc_out.branch(:,RATE_A)<Sline_to) + any(mpc_out.branch(:,RATE_A)<Sline_from);
            % active power line limits could also be considered
            % it is either apparent or active power line limits
            % Constr_Pline = any(mpc_out.branch(:,RATE_A)<abs(results_pf.branch(:,PT)) + any(mpc_out.branch(:,RATE_A)<abs(results_pf.branch(:,PF))
            
            %EIGIL Saving all constraints
            constraint_matrix(n, :, con) = [Constr_PG, Constr_QG, Constr_VG, Constr_Sline];
            % all constraints have to be satisifed for the operating
            % point to be classified as secure
            % keeps the classification for intact and each individual outaged system
            % state
            output_outages(n,con) = (Constr_PG+Constr_QG+Constr_VG+Constr_Sline)~=0;
            % duplicate for parallelization
            output_outages_local(con,1) = (Constr_PG+Constr_QG+Constr_VG+Constr_Sline)~=0;
            
            %Filling out severity matrices
            %Please note, that 'max' corresponds to 'to' and 'min'
            %corresponds to 'from'
            severity_matrix_max(n, idx_constr_PG, con) = (mpc_out.gen(:,PMAX) - results_pf.gen(:,PG))./abs(mpc_out.gen(:,PMAX));
            severity_matrix_max(n, idx_constr_QG, con) = (mpc_out.gen(:,QMAX) - results_pf.gen(:,QG))./abs(mpc_out.gen(:,QMAX));
            severity_matrix_max(n, idx_constr_VM, con) = (mpc_out.bus(:,VMAX)*(1+epsilon) - results_pf.bus(:,VM))./abs(mpc_out.bus(:,VMAX)*(1+epsilon));
            severity_matrix_max(n, idx_constr_Sline(line_ids), con) = (mpc_out.branch(:,RATE_A) - Sline_to)./abs(mpc_out.branch(:,RATE_A));
            
            severity_matrix_min(n, idx_constr_PG, con) = (results_pf.gen(:,PG) - mpc_out.gen(:,PMIN))./abs(mpc_out.gen(:,PMIN));
            severity_matrix_min(n, idx_constr_QG, con) = (results_pf.gen(:,QG) - mpc_out.gen(:,QMIN))./abs(mpc_out.gen(:,QMIN));
            severity_matrix_min(n, idx_constr_VM, con) = (results_pf.bus(:,VM) - mpc_out.bus(:,VMIN)*(1-epsilon))./abs(mpc_out.bus(:,VMIN)*(1-epsilon));
            severity_matrix_min(n, idx_constr_Sline(line_ids), con) = (mpc_out.branch(:,RATE_A) - Sline_from)./abs(mpc_out.branch(:,RATE_A));
            
        else
    
            % power flow failed -- result cannot be trusted
            successful_pf(n,con)=-1;
            fprintf('power flow failed -- the corresponding samples need to be discarded \n');
        end
    end
    % overall output classification
    output(n,1) = sum(output_outages_local)==0;
end

% output composition of dataset
% in general in power systems datasets are often highly unbalanced as only
% small parts of the input space satisify security criteria

fprintf('------finished dataset creation ----- \n')
fprintf('Dataset size:        %6.0f \n',nr_samples);
fprintf('Secure samples:      %6.0f, (%6.3f %% of data samples) \n',sum(output),sum(output)/nr_samples*100);
fprintf('Not secure samples:  %6.0f, (%6.3f %% of data samples)  \n',nr_samples-sum(output),(nr_samples-sum(output))/nr_samples*100);


% Amount of power flows converged

% TODO: check amount of power flows converged
no_unsuccessful_pf = sum(sum(successful_pf == -1));
if no_unsuccessful_pf ~= 0
    fprintf('Number of power flows not converged: %6.0f \n', no_unsuccessful_pf)
else
    fprintf('All power flows converged')
end

% TODO: all the data that did not converge has to be removed
%Find all samples, that did not converged
sample_not_converged = any(successful_pf == -1, 2);

%Remove corresponding output
output(sample_not_converged) = [];
input_lhs(sample_not_converged, :) = [];
input_PD(sample_not_converged, :) = [];
constraint_matrix(sample_not_converged, :, :) = [];
severity_matrix_max(sample_not_converged, :, :) = [];
severity_matrix_min(sample_not_converged, :, :) = [];


%Check if severity-matrix and output matches
output_check_max = any(any(severity_matrix_max < 0, 3), 2);
output_check_min = any(any(severity_matrix_min < 0, 3), 2);
output_check = ~(output_check_min | output_check_max);

if mean(output_check == output) < 1
    error('Error: Severity matrix and constraints doesnt match')
end


% input_lhs + output will be for classification task

%Generating input
if VG_sampling && PD_sampling
    X = [input_lhs, input_PD];
elseif VG_sampling && ~PD_sampling
    X = input_lhs;  
elseif ~VG_sampling && PD_sampling
    X = [input_lhs(:, 1:nr_feat_gen), input_PD];
else
    X = input_lhs(:, 1:nr_feat_gen);
end     

y = output;
    
%save('Classification.mat', 'X', 'y');
%save('Classification2.mat', 'X', 'y')
%save('Classification4_50000.mat', 'X', 'y')
%save('Classification_test_50000.mat', 'X', 'y')
save('classification_val_50000.mat', 'X', 'y')

% input_lhs + active_lf_to/from or apparent_lf_to/from for regression task

%Checking severity matrix

%mean(severity_matrix(:, :, 1) < 0, 1)
%mean(severity_matrix(:, :, 2) < 0, 1)


%EXTRA: Visualize the decision boundary
figure(99)
scatter(input_lhs(logical(output), 1), input_lhs(logical(output), 2), 'b'); hold on
scatter(input_lhs(~logical(output), 1), input_lhs(~logical(output), 2), 'r')
xlabel('Generator 1 power [Pmin, Pmax]', 'FontSize', 14)
ylabel('Generator 2 power [Pmin, Pmax]', 'FontSize', 14)
lgd = legend('Secure', 'Not secure');
lgd.FontSize = 14;
title('N-1 security assesment', 'FontSize', 20)



%Visualize other decisions

%Violating line constraints
for con = 1:nr_con+1
    figure(con)
    subplot(2, 2, 1)
    scatter(input_lhs(constraint_matrix(:, 4, con), 1), input_lhs(constraint_matrix(:, 4, con), 2), 'b', 'MarkerFaceAlpha',.2,'MarkerEdgeAlpha',.2); hold on
    scatter(input_lhs(~constraint_matrix(:, 4, con), 1), input_lhs(~constraint_matrix(:, 4, con), 2), 'r', 'MarkerFaceAlpha',.2,'MarkerEdgeAlpha',.1); hold on
    title('Line constraints')

    %Violating VG constraints
    subplot(2, 2, 2)
    scatter(input_lhs(constraint_matrix(:, 3, con), 1), input_lhs(constraint_matrix(:, 3, con), 2), 'b',  'MarkerFaceAlpha',.2,'MarkerEdgeAlpha',.2); hold on
    scatter(input_lhs(~constraint_matrix(:, 3, con), 1), input_lhs(~constraint_matrix(:, 3, con), 2), 'r',  'MarkerFaceAlpha',.2,'MarkerEdgeAlpha',.1); hold on
    title('VG constraints')

    %Violating BG constraints
    subplot(2, 2, 3)
    scatter(input_lhs(constraint_matrix(:, 1, con), 1), input_lhs(constraint_matrix(:, 1, con), 2), 'b', 'MarkerFaceAlpha',.2,'MarkerEdgeAlpha',.2); hold on
    scatter(input_lhs(~constraint_matrix(:, 1, con), 1), input_lhs(~constraint_matrix(:, 1, con), 2), 'r',  'MarkerFaceAlpha',.2,'MarkerEdgeAlpha',.1); hold on
    title('Bus generation constraints')


    %Violating BG constraints
    subplot(2, 2, 4)
    scatter(input_lhs(any(constraint_matrix(:, :, con), 2), 1), input_lhs(any(constraint_matrix(:, :, con), 2), 2), 'b',  'MarkerFaceAlpha',.2,'MarkerEdgeAlpha',.2); hold on
    scatter(input_lhs(~any(constraint_matrix(:, :, con), 2), 1), input_lhs(~any(constraint_matrix(:, :, con), 2), 2), 'r',  'MarkerFaceAlpha',.2,'MarkerEdgeAlpha',.1); hold on
    title('All constraints together')
end



