% to better understand dataset analyse obtained datasets
clear all;
close all;
% number of buses 
nb = 39;
% load entire workspace
load(strcat('workspace_',num2str(nb)));


% power demand 
mean_input = mean(data_input,2);
std_input = std(data_input,0,2);

% create histograms for quantities of interest
hist(data_input(end,:))