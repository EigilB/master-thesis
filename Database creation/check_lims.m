function [Viol,Viol_mag,success] = check_lims(mpc_out,tol)

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
nb = size(mpc_out.bus,1);
% get number of lines
nl = size(mpc_out.branch,1);
% get number of generators (including synchronous generators)
ng = size(mpc_out.gen,1);


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

% run power flow
try 
    results_pf = runpf(mpc_out,pf_opt);
    % matpower crahes sometimes with enforce_q_lims = true
catch  e %e is an MException struct
        fprintf(1,'The identifier was:\n%s',e.identifier);
        fprintf(1,'There was an error! The message was:\n%s',e.message);
        % more error handling...
    
    % deactivate enforce q lims
    pf_opt.pf.enforce_q_lims = 0;
    results_pf = runpf(mpc_out,pf_opt);
end 
success = results_pf.success;
if success == 0
        
    % deactivate enforce q lims
    pf_opt.pf.enforce_q_lims = 0;
    results_pf = runpf(mpc_out,pf_opt);
end 
success = results_pf.success;

Viol = [-1 -1 -1 -1];
Viol_mag = [-1 -1 -1 -1];

if success == 1
    %             successful_pf(n,con)=1;
    %
    %             % write active power line flows
    %             active_lf_to(n,line_ids,con) = results_pf.branch(:,PT);
    %             active_lf_from(n,line_ids,con) = results_pf.branch(:,PF);
    %
    %             % write apparent power line flows
    %             apparent_lf_to(n,line_ids,con) = (results_pf.branch(:,PF).^2+results_pf.branch(:,QF).^2).^(0.5);
    %             apparent_lf_from(n,line_ids,con) = (results_pf.branch(:,PF).^2+results_pf.branch(:,QF).^2).^(0.5);
    
    
    % check satisfaction of constraints
    % active power generator limits
   % Constr_PG = any((mpc_out.gen(:,PMAX)*(1+tol)+tol)<results_pf.gen(:,PG))+any((mpc_out.gen(:,PMIN)*(1-tol)-tol)>results_pf.gen(:,PG));
    tol_PG = tol * (mpc_out.gen(:,PMAX)-mpc_out.gen(:,PMIN));
    Constr_PG = any((mpc_out.gen(:,PMAX)+tol_PG)<results_pf.gen(:,PG))+any((mpc_out.gen(:,PMIN)-tol_PG)>results_pf.gen(:,PG));
    Constr_PG_max = max(max(results_pf.gen(:,PG)-mpc_out.gen(:,PMAX)),max(mpc_out.gen(:,PMIN)-results_pf.gen(:,PG)));
    % reactive power generator limits
    tol_QG = tol * (mpc_out.gen(:,QMAX)-mpc_out.gen(:,QMIN));
    Constr_QG = any((mpc_out.gen(:,QMAX)+tol_QG)<results_pf.gen(:,QG))+any((mpc_out.gen(:,QMIN)-tol_QG)>results_pf.gen(:,QG));
    Constr_QG_max = max(max(results_pf.gen(:,QG)-mpc_out.gen(:,QMAX)),max(mpc_out.gen(:,QMIN)-results_pf.gen(:,QG)));
    % voltage magnitudes limits
    tol_V =  tol * (mpc_out.bus(:,VMAX)-mpc_out.bus(:,VMIN));
    Constr_V = any((mpc_out.bus(:,VMAX)+tol_V)<results_pf.bus(:,VM))+any((mpc_out.bus(:,VMIN)-tol_V)>results_pf.bus(:,VM));
    Constr_V_max = max(max(results_pf.bus(:,VM)-mpc_out.bus(:,VMAX)),max(mpc_out.bus(:,VMIN)-results_pf.bus(:,VM)));
    % we consider apparent branch flow limits
    % compute apparent branch flows in both directions
    % S^2 = P^2 + Q^2 holds true as S is comples phasor: S = P +
    % jQ
    Sline_to = (results_pf.branch(:,PF).^2+results_pf.branch(:,QF).^2).^(0.5);
    Sline_from = (results_pf.branch(:,PT).^2+results_pf.branch(:,QT).^2).^(0.5);
    Constr_Sline = any((mpc_out.branch(:,RATE_A)*(1+tol))<Sline_to) + any((mpc_out.branch(:,RATE_A)*(1+tol))<Sline_from);
    Constr_Sline_max = max(max(Sline_to-mpc_out.branch(:,RATE_A)),max(Sline_from-mpc_out.branch(:,RATE_A)));
    % active power line limits could also be considered
    % it is either apparent or active power line limits
    % Constr_Pline = any(mpc_out.branch(:,RATE_A)<abs(results_pf.branch(:,PT)) + any(mpc_out.branch(:,RATE_A)<abs(results_pf.branch(:,PF))
    
    Viol = [Constr_PG Constr_QG Constr_V Constr_Sline];
    Viol_mag = [Constr_PG_max Constr_QG_max Constr_V_max Constr_Sline_max];
    
else
    
    % power flow failed -- result cannot be trusted
    success=-1;
    fprintf('power flow failed -- the corresponding samples need to be discarded \n');
end

end

