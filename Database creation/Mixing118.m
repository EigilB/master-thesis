% Mixing Big2 and Big2
cd '/Users/Eigil/Dropbox/DTU/Speciale/Data Generation'



load_var2 = load('Big2.mat'); 
X2 = load_var2.X;
y2 = load_var2.y;

mean(y2)

load_var3  = load('Big3.mat');
X3 = load_var3.X;
y3 = load_var3.y;

sum(y3 == 1)

safe_indices = (y3 == 1);
y3_safe = y3(safe_indices);
X3_safe = X3(safe_indices, :);
y3_unsafe = y3(~safe_indices);
X3_unsafe = X3(~safe_indices, :);

%Number of safe points in total
no_safe_points = sum(y2 == 1) + sum(safe_indices);
no_sampled_unsafe_points = no_safe_points - sum(~(y2 == 1))

idx = randperm(length(y3_unsafe));
idx = idx(1:no_sampled_unsafe_points);

y3_sampled_unsafe = y3_unsafe(idx);
X3_sampled_unsafe = X3_unsafe(idx, :);

size(y3_sampled_unsafe)
size(X3_sampled_unsafe)

%Collecting everything together
X_total = [X2; X3_safe; X3_sampled_unsafe];
y_total = [y2; y3_safe; y3_sampled_unsafe];

size(y_total)

%Keeping track of indices from each sampling technique and shuffling
%First le
no_procedure1 = length(y2);
no_procedure2 = length(y_total) - no_procedure1;


shuffling_idx = randperm(length(y_total));
idx_procedure1 = shuffling_idx(1:no_procedure1);
idx_procedure2 = shuffling_idx((no_procedure1+1):end);

X_total_shuffled = X_total(shuffling_idx, :);
y_total_shuffled = y_total(shuffling_idx);

X = X_total_shuffled;
y = y_total_shuffled;

%save('BigCombined1.mat', 'X', 'y', 'idx_procedure1', 'idx_procedure2')



