function[H,Z,Sbar,A,Q,labels,converge_Z,converge_F,converge_Z_G] = Train_4new(X, cls_num, anchor, alpha, gamma, delta)
% X is a cell data, each cell is a matrix in size of d_v *N,each column is a sample;
% cls_num is the clustering number
% anchor is the anchor number
% alpha,gamma and delta are the parameters

nV = length(X);
N = size(X{1},2);
t = anchor;
nC = cls_num;

%% ============================ Initialization ============================
% Pre-allocate cell arrays for efficiency
Z = cell(1, nV);
Q = cell(1, nV);
A = cell(1, nV);
E = cell(1, nV);
F = cell(1, nV);
Y = cell(1, nV);
YY = cell(1, nV);
new1 = cell(1, nV);
J = cell(1, nV + 1);
W = cell(1, nV + 1);

for k = 1:nV
    Z{k} = zeros(t, N);
    Q{k} = eye(t,t); % Initialize Q as identity for better numerical stability
    A{k} = zeros(size(X{k},1), t);
    E{k} = zeros(size(X{k},1), N);
    F{k} = zeros(size(X{k},1), N);
    Y{k} = zeros(size(X{k},1), N);
    YY{k} = zeros(size(X{k},1), N);
    new1{k} = zeros(t, N);
end
sX = [t, N, nV+1];

H = zeros(t, N); % Initialize H
% If H needs a specific non-zero initialization, uncomment/modify this:
% H(:, 1:min(t, N)) = eye(t, N); % Initialize with identity where possible

for i = 1:nV + 1
    J{i} = zeros(t, N);
    W{i} = zeros(t, N);
end

Isconverg = 0;
epson = 1e-7; % Convergence tolerance
iter = 0;
mu = 0.0001;
max_mu = 10e12;
pho_mu = 2; % rho_mu in ADMM literature

converge_Z = [];
converge_F = [];
converge_Z_G = [];

%% =============================== ADMM Loop ===============================
while(Isconverg == 0)
    iter = iter + 1;

    %% ============================== Update A^v =============================
    for i = 1:nV
        % Combined term for A update. Consider if this truly reflects your objective.
        % For faster matrix multiplication, ensure dimensions are optimal.
        % (Size(X{i},1) x t) = (Size(X{i},1) x N) * (N x t) * (t x t)
        % (Size(X{i},1) x N) * (N x t)
        A_temp_part1 = (YY{i} + mu * X{i} - mu * F{i}) * (H' * Q{i}');
        A_temp_part2 = (Y{i} + mu * X{i} - mu * E{i}) * Z{i}';
        A_temp = A_temp_part1 + A_temp_part2;
        
        [Au,~,Av] = svd(A_temp,'econ');
        A{i} = Au * Av';
    end

   % ============================== Update Q^v =============================
    for i = 1:nV
        % (t x t) = (t x d_v) * (d_v x N) * (N x t) + (t x N) * (N x t)
        Q_temp = A{i}' * (X{i} - F{i} + YY{i}/mu) * H' + Z{i}*( J{i} - W{i}/mu)';
        [Qu,~,Qv] = svd(Q_temp,'econ');
        Q{i} = Qu * Qv';
    end

    %% =========================== Update E^v, F^v ===========================
    for i = 1:nV
      [E{i}] = solve_l1l2(X{i} - A{i} * Z{i} + Y{i} / mu, gamma / mu);
      [F{i}] = solve_l1l2(X{i} - A{i} * Q{i} * H + YY{i} / mu, alpha / mu);
    end

    %% ============================ Update Z^k ============================
    % Pre-calculate common term for efficiency outside the loop if possible,
    % but here it depends on A{k}, so inside is fine.
    inv_tmp = (1 / (2 * mu)) * eye(t, t); % Direct scalar multiplication is faster than inv(matrix)
    for k = 1:nV
        Z_temp = mu * A{k}' * X{k} - mu * A{k}' * E{k} + A{k}' * Y{k} + mu * Q{k} * J{k} - Q{k} * W{k};
        Z{k} = inv_tmp * Z_temp; % Optimized
        
        % Project each column to simplex
        for col_idx = 1:N
            Z{k}(:, col_idx) = EProjSimplex_new(Z{k}(:, col_idx));
        end
    end

    %% ============================== Update H =============================
    % FIX: Accumulate contributions from all views correctly
    H_numerator_sum = zeros(t, N); % Initialize accumulator for numerator
    for i = 1:nV
        H_numerator_sum = H_numerator_sum + (mu * Q{i}' * A{i}' * X{i} - mu * Q{i}' * A{i}' * F{i} + Q{i}' * A{i}' * YY{i});
    end
    
    % Corrected H update
    H_temp = (1 / ((nV + 1) * mu)*eye(t)) * (H_numerator_sum + mu * J{nV+1} - W{nV+1});
    % Project each column to simplex
    for col_idx = 1:N
        H(:, col_idx) = EProjSimplex_new(H_temp(:, col_idx));
    end

    %% ============================= Update J^k ==============================
    for i = 1:nV
        new1{i} = Q{i}' * Z{i};
    end
    t_Z = [new1, {H}]; % All matrices stacked into a cell array

    % Convert cell array to tensor
    Z_tensor = cat(3, t_Z{:});
    W_tensor = cat(3, W{:}); % Use {:} for direct concatenation

    % Solve for J_tensor using solve_G
    J_tensor = solve_G(Z_tensor + (1/mu) * W_tensor, mu, sX, delta); % Assuming sX is correctly defined as [t, N, nV+1]
    
    % Distribute J_tensor back to J cell array
    for k_tensor = 1:(nV + 1)
        J{k_tensor} = J_tensor(:,:,k_tensor);
    end

    %% ============================== Update W ===============================
    % Update W_tensor directly from the difference
    W_tensor = W_tensor + mu * (Z_tensor - J_tensor);

    % Distribute W_tensor back to W cell array
    for k_tensor = 1:(nV + 1)
        W{k_tensor} = W_tensor(:,:,k_tensor);
    end

    %% ========================== Update Y, YY ==========================
    for i = 1:nV
        Y{i} = Y{i} + mu * (X{i} - A{i} * Z{i} - E{i});
        YY{i} = YY{i} + mu * (X{i} - A{i} * Q{i} * H - F{i});
    end

    %% ===================== Checking Converge Condition =====================
    max_Z = 0;
    max_Z_F = 0;
    max_Z_G = 0;
    Isconverg = 1; % Assume convergence until a condition fails

    for k = 1:nV
        % Check primal residuals
        if norm(X{k} - A{k} * Z{k} - E{k}, 'inf') > epson
            max_Z = max(max_Z, norm(X{k} - A{k} * Z{k} - E{k}, 'inf'));
            Isconverg = 0;
        end
        if norm(X{k} - A{k} * Q{k} * H - F{k}, 'inf') > epson
            max_Z_F = max(max_Z_F, norm(X{k} - A{k} * Q{k} * H - F{k}, 'inf'));
            Isconverg = 0;
        end
    end
    
    % Check residual for Z_tensor and J_tensor
    for k_tensor = 1:(nV + 1)
        if norm(t_Z{k_tensor} - J{k_tensor}, 'inf') > epson
            max_Z_G = max(max_Z_G, norm(t_Z{k_tensor} - J{k_tensor}, 'inf'));
            Isconverg = 0;
        end
    end

    converge_Z = [converge_Z max_Z];
    converge_F = [converge_F max_Z_F];
    converge_Z_G = [converge_Z_G max_Z_G];
    
    % Force stop condition - consider a more robust one for final work
    if iter > 19
        Isconverg = 1;
    end
    
    mu = min(mu * pho_mu, max_mu); % Update mu for next iteration
end

%% =========================== Final Processing ===========================
%tt_Z = [Z, {H}];

% Sbar = [];
% for i = 1:(nV +1)
%     Sbar = cat(1, Sbar, (1 / sqrt(nV + 1)) * (t_Z{i}));
% end


Sbar = [];
for i = 1:(nV)
    Sbar = cat(1, Sbar, (1 / sqrt(nV)) * (Z{i}));
end

% Note: If nC < t, SVD truncates. If nC > t, padding occurs or it's an error.
% Ensure nC is a reasonable value relative to t.
[U, ~, ~] = mySVD(Sbar', nC); % Only need U for kmeans

%rng('shuffle'); % Use 'shuffle' for better randomness across runs
rand('twister',5489)

labels = litekmeans(U, nC, 'MaxIter', 100, 'Replicates', 10);

end