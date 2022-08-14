addpath(genpath("tapas"));
config = tapas_hgf_binary_pu_config();
unitsq_sgm_config = tapas_unitsq_sgm_config();
optim_config = tapas_quasinewton_optim_config();
u = [zeros(160,1);ones(160,1)];

% get optimal estimate
opt_est = tapas_fitModel([], u, config, 'tapas_bayes_optimal_binary_config');

% set config to optimal estimates
config.ommu=opt_est.p_prc.om;
config.logalmu=log(opt_est.p_prc.al);
config.eta0mu=opt_est.p_prc.eta0;
config.eta1mu=opt_est.p_prc.eta1;
config.logkamu=log(opt_est.p_prc.ka);
config.mu_0mu=opt_est.p_prc.mu_0;
config.rhomu=opt_est.p_prc.rho;
config = tapas_align_priors(config);

% change optimal to ASD by increasing alpha
asd_p = opt_est.p_prc.p;
asd_p(15)=0.6;


asd_sim = tapas_simModel(u,'tapas_hgf_binary_pu',asd_p,'tapas_unitsq_sgm',...
    10,123456789);
sim = tapas_simModel(u,'tapas_hgf_binary_pu',opt_est.p_prc.p,'tapas_unitsq_sgm',...
    10,123456789);

sim_fit = tapas_fitModel(sim.y,u, config,unitsq_sgm_config ,optim_config);
asd_fit = tapas_fitModel(asd_sim.y,u, tapas_hgf_binary_pu_config(), unitsq_sgm_config ,optim_config);


% create different individuals to bootstrap the alpha and omega3 values
num_bootstrap = 100;
alphas = zeros(num_bootstrap,2);
omegas3 = zeros(num_bootstrap,2);
omegas2 = zeros(num_bootstrap,2);
parfor i=1:num_bootstrap
    rng(i);
    asd_sim = tapas_simModel(u,'tapas_hgf_binary_pu',asd_p,'tapas_unitsq_sgm', 10);
    sim = tapas_simModel(u,'tapas_hgf_binary_pu',opt_est.p_prc.p,'tapas_unitsq_sgm', 10);

    sim_fit = tapas_fitModel(sim.y,u, config,unitsq_sgm_config ,optim_config);
    asd_fit = tapas_fitModel(asd_sim.y,u, tapas_hgf_binary_pu_config(), unitsq_sgm_config ,optim_config);
    alphas(i,:) = [asd_fit.p_prc.al,sim_fit.p_prc.al];
    omegas3(i,:) = [asd_fit.p_prc.om(3),sim_fit.p_prc.om(3)];
    omegas2(i,:) = [asd_fit.p_prc.om(2),sim_fit.p_prc.om(2)];
end
save 'hgf.mat' asd_fit sim_fit alphas omegas3
%% CI

disp("NT omega 2")
disp(bootci(100000, @(d)(median(d)) ,omegas2(:,2)))
disp("NT omega 3")
disp(bootci(100000, @(d)(median(d)) ,omegas3(:,2)))
disp("NT alphas")
disp(bootci(100000, @(d)(median(d)) ,alphas(:,2)))

disp("ASD omega 2")
disp(bootci(100000, @(d)(median(d)) ,omegas2(:,1)))
disp("ASD omega 3")
disp(bootci(100000, @(d)(median(d)) ,omegas3(:,1)))
disp("ASD alphas")
disp(bootci(100000, @(d)(median(d)) ,alphas(:,1)))
