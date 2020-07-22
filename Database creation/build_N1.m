function mpc = build_N1(mpc,con_list,ID_gen,v_IDs)
% this function creates duplicates of the intact system state in the mpc
% struct with the corresponding line outage
% These duplicates are treated as different systems (over 'area' column)
% then

%% define named indices into data matrices
[PQ, PV, REF, NONE, BUS_I, BUS_TYPE, PD, QD, GS, BS, BUS_AREA, VM, ...
    VA, BASE_KV, ZONE, VMAX, VMIN, LAM_P, LAM_Q, MU_VMAX, MU_VMIN] = idx_bus;
[F_BUS, T_BUS, BR_R, BR_X, BR_B, RATE_A, RATE_B, RATE_C, ...
    TAP, SHIFT, BR_STATUS, PF, QF, PT, QT, MU_SF, MU_ST, ...
    ANGMIN, ANGMAX, MU_ANGMIN, MU_ANGMAX] = idx_brch;
[GEN_BUS, PG, QG, QMAX, QMIN, VG, MBASE, GEN_STATUS, PMAX, PMIN, ...
    MU_PMAX, MU_PMIN, MU_QMAX, MU_QMIN, PC1, PC2, QC1MIN, QC1MAX, ...
    QC2MIN, QC2MAX, RAMP_AGC, RAMP_10, RAMP_30, RAMP_Q, APF] = idx_gen;


bus_orig = mpc.bus;
gen_orig = mpc.gen;
branch_orig = mpc.branch;
gencost_orig = mpc.gencost;
ng = size(gen_orig,1);
nb = size(bus_orig,1);
nl = size(branch_orig,1);

nr_con = size(con_list,1);

for c = 1:nr_con
    bus = bus_orig;
    branch = branch_orig;
    gen = gen_orig;
    
    % add +nb to bus index
    bus(:,1) = bus_orig(:,1) + c*nb;
    % outaged system state is treated as an additional area (so MATPOWER
    % does not connect them)
    
    bus(:,BUS_AREA) = c+1;
    % add +nb to branch indices
    branch(:,F_BUS) = branch_orig(:,F_BUS) + c*nb;
    branch(:,T_BUS) = branch_orig(:,T_BUS) + c*nb;
    
    % remove the branch
    branch(con_list(c),:) =[];
    
    % add +nb to generator indices
    gen(:,GEN_BUS) = gen_orig(:,GEN_BUS) + c*nb;
    
    mpc.bus = [mpc.bus; bus];
    mpc.branch = [mpc.branch; branch];
    mpc.gen = [mpc.gen; gen];
    gencost_orig(:,5) = 0;
    gencost_orig(:,6) = 0;
    gencost_orig(:,7) = 0;
    mpc.gencost = [mpc.gencost; gencost_orig];
end


% add user defined constraint to the opf which fixes the generator active
% power and voltages

%pg_IDs=[2,3];

cost_params.ng =ng;
cost_params.ID_gen = ID_gen;
cost_params.v_IDs = v_IDs;
cost_params.nb = nb;
cost_params.nc = nr_con;

% preventive SCOPF formulation
% active power generator are fixed, and generator voltage set-points as
% well
% note that the active power generator slack must be allowed to vary to compensate the change in
% losses
mpc=add_userfcn(mpc, 'formulation', @linkage_constraints,cost_params);





end 


%%-----  formulation  --------------------------------------------------
function om = linkage_constraints(om, cost_params)

%% define named indices into data matrices
[PQ, PV, REF, NONE, BUS_I, BUS_TYPE, PD, QD, GS, BS, BUS_AREA, VM, ...
    VA, BASE_KV, ZONE, VMAX, VMIN, LAM_P, LAM_Q, MU_VMAX, MU_VMIN] = idx_bus;
[F_BUS, T_BUS, BR_R, BR_X, BR_B, RATE_A, RATE_B, RATE_C, ...
    TAP, SHIFT, BR_STATUS, PF, QF, PT, QT, MU_SF, MU_ST, ...
    ANGMIN, ANGMAX, MU_ANGMIN, MU_ANGMAX] = idx_brch;
[GEN_BUS, PG, QG, QMAX, QMIN, VG, MBASE, GEN_STATUS, PMAX, PMIN, ...
    MU_PMAX, MU_PMIN, MU_QMAX, MU_QMIN, PC1, PC2, QC1MIN, QC1MAX, ...
    QC2MIN, QC2MAX, RAMP_AGC, RAMP_10, RAMP_30, RAMP_Q, APF] = idx_gen;


ng= cost_params.ng;
ID_gen = cost_params.ID_gen;
v_IDs = cost_params.v_IDs;
nb = cost_params.nb;
nr_con = cost_params.nc;


npg = size(ID_gen,1);

L_PG = zeros(npg*nr_con,1);
U_PG = zeros(npg*nr_con,1);

A_PG = zeros(npg*nr_con,ng*(nr_con+1));

for g = 1:npg
    for c=1:nr_con
        A_PG(g+(c-1)*npg,ID_gen(g)) = 1;
        A_PG(g+(c-1)*npg,ID_gen(g)+ng*c) = -1;
    end
end

om = add_constraints(om, 'Linkage_Constraints_PG', A_PG, L_PG, U_PG, {'Pg'});


nvg =size(v_IDs,1);

L_VG = zeros(nvg*nr_con,1);
U_VG = zeros(nvg*nr_con,1);

A_VG = zeros(nvg*nr_con,nb*(nr_con+1));

for g = 1:nvg
    for c=1:nr_con
        A_VG(g+(c-1)*nvg,v_IDs(g)) = 1;
        A_VG(g+(c-1)*nvg,v_IDs(g)+nb*c) = -1;
    end
end

om = add_constraints(om, 'Linkage_Constraints_VG', A_VG, L_VG, U_VG, {'Vm'});

% om = add_costs(om, 'GeneratorVoltageCost', struct('H',H,'Cw', Cw,'mm',m,'rh',rhat), {'Vm','Pg'});

end



