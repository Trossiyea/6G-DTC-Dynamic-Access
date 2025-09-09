X_file = 'Data.mat';
%% ====== 1) 分别读取 X_true / C_true ======
% 读取 X_true（兼容文件中变量名不叫 X_true 的情况）
Sx = load(X_file);
fnx = fieldnames(Sx);
X_true = Sx.(fnx{1});   % 若文件里变量名不同，这里取到第一个变量


% 基本信息
if ndims(X_true) < 3
    error('X_true 需要是 3D (I x J x K)。当前尺寸: %s', mat2str(size(X_true)));
end
[I,J,K] = size(X_true);
fprintf('Loaded X_true: %d x %d x %d\n', I, J, K);

%% ====== 2) 可视化 1：指定频切片的 dB 等高填色图 ======
k = min(20, K);          % 选一个存在的切片索引（可改）
eps_db = 1e-10;          % 防止 log10(0)
map_db = 10*log10(max(X_true(:,:,k),0) + eps_db);

figure;
contourf(map_db, 100, 'LineStyle','none');
axis image; colorbar; colormap jet;
title(sprintf('True Map (k=%d) [dBm]', k));
xlabel('X'); ylabel('Y');