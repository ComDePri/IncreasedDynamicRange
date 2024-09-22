%% Path to raw_data.mat downloaded from https://osf.io/83wnu/
PATH = 'raw_data.mat';
%%
load(PATH)
nt_idx = subject_info.Group==1;
dys_idx = subject_info.Group==2;
asd_idx = subject_info.Group==3;

titles = ["NT","DYS","ASD"];
idx = [nt_idx,dys_idx,asd_idx];
for i=1:3
    e = tapping_data.e(idx(:,i),:,:);
    r = tapping_data.r(idx(:,i),:,:);
    s = tapping_data.s(idx(:,i),:,:);
    save(titles(:,i), "e","r","s");
end
