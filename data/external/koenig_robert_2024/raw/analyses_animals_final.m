%%%%%
clearvars

figure_dir = './Rstudio_traj/figures/';

load("trajectories-animals.mat")

number_perm_model=10000;
number_perm_cluster=10000;

% remove bad subjects and trials
TT=TT(TT.include==1,:);
%TT=TT(TT.include==1 & TT.correct==1,:); to run on correct trials only

%renumber subjects
[~,~,TT.subject] = unique(TT.subject);

TT.xposi=TT.xposi.*(-1);

% compute execution time RDM
threshold_x = 270;
threshold_y = 380;
[~,TT.execution_time]=max(abs(TT.xposi)>threshold_x & abs(TT.yposi)>threshold_y,[],2);
TT.execution_time(TT.execution_time==1) = NaN;
R=[];
for p=1:max(TT.subject) %each separatly to give the same weight
    for i = 1:27 % nbr of stimuli
        int=TT.execution_time(TT.subject==p & TT.sortedstimulusnr==i,:);
        R(p,i)=nanmean(int,1);
    end
end
% reorder the stimuli from OSF
idx_stim = [10 11 12 13 14 15 16 17 18 1 2 3 4 5 6 7 8 9 19 20 21 22 23 24 25 26 27]; % correct order

RDM_exec=pdist(nanmean(R,1)');
RDM_exec=squareform(RDM_exec);
x_plot=figure(1);
rdm=RDM_exec;
loweridx = find(tril(ones(size(rdm,1)),-1));
rdm2 = normalize(rdm(loweridx),'range');
imagesc(squareform(rdm2))
set(gca,'TickDir', 'none')
axis equal
ylim([0.5 27.5])
xlim([0.5 27.5])
colormap magma
x_plot.Units = 'inches';
x_plot.OuterPosition = [1.5 1.5 5 5];
exportgraphics(x_plot,[figure_dir 'RDM_exec_exp2.tif'],'resolution',300)


% load similarity judgements
load('Data.mat')
RDM_behav=OSF.behaviData;
% reorder the stimuli
RDM_behav=RDM_behav(idx_stim,idx_stim);
x_plot=figure(1);
rdm=RDM_behav;
loweridx = find(tril(ones(size(rdm,1)),-1));
rdm2 = normalize(rdm(loweridx),'range');
imagesc(squareform(rdm2))
set(gca,'TickDir', 'none')
axis equal
ylim([0.5 27.5])
xlim([0.5 27.5])
colormap magma
x_plot.Units = 'inches';
x_plot.OuterPosition = [1.5 1.5 5 5];
exportgraphics(x_plot,[figure_dir 'RDM_behav_exp2.tif'],'resolution',300)











% compute mean trajectory for each stimulus and participant
M=[];
for p=1:max(TT.subject) %each separtly to give the same weight
    for i = 1:27 % nbr of stimuli
        int=TT.xposi(TT.subject==p & TT.sortedstimulusnr==i,:);
        M(p,i,:)=nanmean(int,1);
    end
end

My=[];
for p=1:max(TT.subject) %each separately to give the same weight
    for i = 1:27 % nbr of stimuli
        int=TT.yposi(TT.subject==p & TT.sortedstimulusnr==i ,:);
        My(p,i,:)=nanmean(int,1);
    end
end

% average across participants and plot
MM=M; MMy=My;
M=squeeze(nanmean(M,1));
My=squeeze(nanmean(My,1));

%%% individuals trials example
idx_part=[3 63 83];
co = repelem(brewermap(3,'Set2'),1,1); %96 objects
co(:,4)=0.8;

for ee=1:3
    int_TT=TT(TT.subject==idx_part(ee),:);
    close all; x_plot=figure(1);
    hold on
    for i=1:108 %size(int_TT,1)
        if int_TT.sortedstimulusnr(i)<10
            plot(int_TT.xposi(i,:),int_TT.yposi(i,:),'color',co(1,:),'linewidth',1.2)
        elseif int_TT.sortedstimulusnr(i)<19
            plot(int_TT.xposi(i,:),int_TT.yposi(i,:),'color',co(2,:),'linewidth',1.2)
        elseif int_TT.sortedstimulusnr(i)<28
            plot(int_TT.xposi(i,:),int_TT.yposi(i,:),'color',co(3,:),'linewidth',1.2)
        end
    end
    ylim([0 600])
    xlim([-600 600])
    axis equal
    exportgraphics(x_plot,[figure_dir 'Mean_traj_exp2_' num2str(ee) '.tif'],'resolution',300)
end

%%%%% Mean trajectories
co = repelem(brewermap(3,'Set2'),1,1); %27 objects
co(:,4)=0.8;
close all; x_plot=figure(1);

plot(zeros(511,1),-5:505,'k:', LineWidth=1)
hold on
x1=plot(M(1:9,:)',My(1:9,:)','color',co(1,:),'linewidth',0.8);
x2=plot(M(10:18,:)',My(10:18,:)','color',co(2,:),'linewidth',0.8);
x3=plot(M(19:27,:)',My(19:27,:)','color',co(3,:),'linewidth',0.8);

hleglines = [x1(1) x2(1) x3(1)];
ylim([0 500])
xlim([-330 330])
axis equal
xlabel('Horizontal position (pixels)')
ylabel('Vertical position (pixels)')
[~, hobj, ~, ~] =legend(hleglines,{'Animal','Lookalike','Object'},'Location','southwest');legend boxoff
hl = findobj(hobj,'type','line');
set(hl,'LineWidth',4);
ht = findobj(hobj,'type','text');
set(ht,'FontSize',14);
set(gca, 'FontSize',14,'XTick',[-300 0 300],'YTick',[0 250 500])

x_plot.Units = 'inches';
x_plot.OuterPosition = [1.5 1.5 5 5];
exportgraphics(x_plot,[figure_dir 'Mean_traj_exp2.tif'],'resolution',300)

%%%%% figure horizontal position
co = repelem(brewermap(3,'Set2'),1,1); %27 objects
co(:,4)=0.8;
close all; x_plot=figure(1);
plot(0:800,zeros(801,1),'k:', LineWidth=1)
hold on
x1=plot(0:800,M(1:9,:),'color',co(1,:),'linewidth',0.8);
x2=plot(0:800,M(10:18,:),'color',co(2,:),'linewidth',0.8);
x3=plot(0:800,M(19:27,:),'color',co(3,:),'linewidth',0.8);

hleglines = [x1(1) x2(1) x3(1)];
xlim([0 800])
ylim([-330 330])
ylabel('Horizontal position (pixels)')
xlabel('Movement time (ms)')
[~, hobj, ~, ~] =legend(hleglines,{'Animal','Lookalike','Object'},'Location','southwest');legend boxoff
hl = findobj(hobj,'type','line');
set(hl,'LineWidth',4);
ht = findobj(hobj,'type','text');
set(ht,'FontSize',14);
set(gca, 'YDir','reverse','FontSize',14,'YTick',[-300 0 300])

x_plot.Units = 'inches';
x_plot.OuterPosition = [1.5 1.5 6 5];
exportgraphics(x_plot,[figure_dir 'Mean_xpos_exp2.tif'],'resolution',300)

% compute RDMs trajectories
for t=1:size(M,2)
    M_RDM_traj(:,t)=pdist(M(:,t));
    int=squareform(pdist(M(:,t)));
    M_RDM_traj_sub_1(:,t)=reshape(int(10:18,1:9),[],1);
    M_RDM_traj_sub_2(:,t)=reshape(int(19:27,1:9),[],1);
    M_RDM_traj_sub_3(:,t)=reshape(int(19:27,10:18),[],1);
end

% visualisation over time
close all; figure(1)
x_plot=figure(1);
for t=1:50:size(M_RDM_traj,2)
    hold on

    rdm = squeeze(squareform(M_RDM_traj(:,t)));
    loweridx = find(tril(ones(size(rdm,1)),-1));
    rdm2 = normalize(rdm(loweridx),'range');
    imagesc(squareform(rdm2))
    imagesc(squareform(M_RDM_traj(:,t)))
    set(gca, 'YDir','reverse','TickDir', 'none')
    axis equal
    ylim([0.5 27.5])
    xlim([0.5 27.5])
    colormap magma
    x_plot.Units = 'inches';
    x_plot.OuterPosition = [1.5 1.5 5 5];
    exportgraphics(x_plot,[figure_dir 'RDM_traj_exp2_' num2str(t-1) '.tif'],'resolution',300)
end

close all
x_plot=figure(1);
co = repelem(brewermap(3,'Set2'),9,1); %27 objects
mm=50;
R = movmean(M_RDM_traj,mm,2);
for t=1:25:size(M_RDM_traj,2)
    rdm = squareform((R(:,t)));
    rng(1)
    Y = mdscale(rdm,2,'Criterion','stress');
    scatter(Y(:,1),Y(:,2),140,co,'filled','MarkerFaceAlpha',.8,'MarkerEdgeAlpha',.8)
    ax=gca;
    ax.XColor='w';
    ax.YColor='w';
    x_plot.Units = 'inches';
    x_plot.OuterPosition = [1.5 1.5 5 5];
    exportgraphics(x_plot,[figure_dir 'MDS_traj_exp2_' num2str(t-1) '.tif'],'resolution',600)
end













%%%% RSA with the models
model_animacy=ones(27,27);
model_animacy(1:9,1:9)=0;
model_animacy(10:27,10:27)=0;

model_appearance=ones(27,27);
model_appearance(1:18,1:18)=0;
model_appearance(19:27,19:27)=0;

models = {model_animacy, model_appearance};
filenames = {'RDM_model_animal_exp2.tif', 'RDM_model_lookalike_exp2.tif'};
for i=1:2
    x_plot=figure(1);
    imagesc(models{i});
    set(gca, 'TickDir', 'none')
    axis equal
    ylim([0.5 27.5])
    xlim([0.5 27.5])
    colormap magma
    x_plot.Units = 'inches';
    x_plot.OuterPosition = [1.5 1.5 5 5];
    exportgraphics(x_plot,[figure_dir,filenames{i}],'resolution',300)
end

RSA_animacy=corr(M_RDM_traj,squareform(model_animacy)',"type","Spearman")';
RSA_appearance=corr(M_RDM_traj,squareform(model_appearance)',"type","Spearman")';

RSApp_animal=[];
RSApp_appearance=[];

for pp=1:number_perm_model
    idx_subject=randsample(size(MM,1),size(MM,1),1);
    Mpp=squeeze(nanmean(MM(idx_subject,:,:),1));
    Mpp_RDM_traj=[];
    for t=1:size(Mpp,2)
        Mpp_RDM_traj(:,t)=pdist(Mpp(:,t));
    end
    RSApp_animal(pp,:)=corr(Mpp_RDM_traj,squareform(model_animacy)',"type","Spearman")';
    RSApp_appearance(pp,:)=corr(Mpp_RDM_traj,squareform(model_appearance)',"type","Spearman")';
    pp
end

% plot
error_animal=prctile(RSApp_animal,[5 95]); error_animal=error_animal([2,1],:);error_animal=error_animal-mean(RSApp_animal);
error_appearance=prctile(RSApp_appearance,[5 95]);error_appearance=error_appearance([2,1],:); error_appearance=error_appearance-mean(RSApp_appearance);
error_diff=prctile(RSApp_animal-RSApp_appearance,[2.5 97.5]);
co1=brewermap(9,'Blues');
co2=brewermap(9,'Reds');
co = [co1(5,:);co2(5,:)];
x=0:800;

close all
x_plot=figure(1); hold on
yline(0,LineWidth=0.8,LineStyle=":")
shadedErrorBar(x, RSA_animacy,abs(error_animal),'lineprops',{'color',co(1,:),'linewidth',2})
shadedErrorBar(x, RSA_appearance,abs(error_appearance),'lineprops',{'color',co(2,:),'linewidth',2})

t_animacy=find(error_animal(2,:)+RSA_animacy>0);
plot(t_animacy,ones(1,length(t_animacy))*(-0.16),'color',co(1,:),marker=".",MarkerSize=13,LineStyle="none")
t_appearance=find(error_appearance(2,:)+RSA_appearance>0);
plot(t_appearance,ones(1,length(t_appearance))*(-0.20),'color',co(2,:),marker=".",MarkerSize=13,LineStyle="none")
t_diff=find(error_diff(2,:)<0|error_diff(1,:)>0);
plot(t_diff,ones(1,length(t_diff))*(-0.24),'color',[0.7,0.7,0.7],marker=".",MarkerSize=13,LineStyle="none")

set(gca,'TickDir','out');
[~, hobj, ~, ~] =legend({'','Animal model','Lookalike model'},'Location','northwest');legend boxoff
hl = findobj(hobj,'type','line');
set(hl,'LineWidth',4);
ht = findobj(hobj,'type','text');
set(ht,'FontSize',16);
set(gca, 'FontSize',16,'YTick',[0 0.5 1])
xlabel('Movement time (ms)')
ylabel('Spearman correlation')
xlim([0 800])
ylim([-0.28 1])
x_plot.Units = 'inches';
x_plot.OuterPosition = [1.5 1.5 25*0.4 10*0.6];
exportgraphics(x_plot,[figure_dir 'RSA_traj_exp2.tif'],'resolution',600)

%%%% RSA with EEG
M1=load('ResultsStats_rsa_task_Both_PerSubj');
X=[];
for p=1:30
    X(p,:,:,:)=M1.dataPerSubj{1, p}.matrix_decoding;
end
% reorder the stimuli
X=X(:,idx_stim,idx_stim,:);








% visualisation over time
mt=-200:4:500;
N=1; %%% find closest coresponding timepoint
for t=70:90:500
    idx_int=find(abs(mt-t)==min(abs(mt-t)));
    idx_time(N)=idx_int(1);
    N=N+1;
end

%%
close all
x_plot=figure(1);
for t=1:length(idx_time)
    hold on
    rdm = squeeze(mean(X(:,:,:,idx_time(t)),1));
    loweridx = find(tril(ones(size(rdm,1)),-1));
    rdm2 = normalize(rdm(loweridx),'range');
    imagesc(squareform(rdm2))
    set(gca, 'YDir','reverse','TickDir', 'none')
    axis equal
    ylim([0.5 27.5])
    xlim([0.5 27.5])
    colormap magma
    x_plot.Units = 'inches';
    x_plot.OuterPosition = [1.5 1.5 5 5];
    exportgraphics(x_plot,[figure_dir 'RDM_EEG_exp2_' num2str(70+(t-1)*90) '.tif'],'resolution',300)
    drawnow
end

N=1;
for t=70:45:500
    idx_int=find(abs(mt-t)==min(abs(mt-t)));
    idx_time(N)=idx_int(1);
    N=N+1;
end

close all
x_plot=figure(1);
co = repelem(brewermap(3,'Set2'),9,1); %27 objects
mm=50;
for t=1:length(idx_time)
    rdm = squeeze(mean(X(:,:,:,idx_time(t)),1));
    rng(1)
    Y = mdscale(rdm,2,'Criterion','stress');
    scatter(Y(:,1),Y(:,2),140,co,'filled','MarkerFaceAlpha',.8,'MarkerEdgeAlpha',.8)
    ax=gca;
    ax.XColor='w';
    ax.YColor='w';
    x_plot.Units = 'inches';
    x_plot.OuterPosition = [1.5 1.5 5 5];
    exportgraphics(x_plot,[figure_dir 'MDS_EEG_exp2_' num2str(70+(t-1)*45) '.tif'],'resolution',600)
end

%% organise all EEG RDMs
RDM_EEG=[];RDM_EEG_sub_1=[];RDM_EEG_sub_2=[];RDM_EEG_sub_3=[];

for p=1:size(X,1)
    for t=1:176
        RDM_EEG(p,:,t)=squareform(squeeze(X(p,:,:,t)));
        RDM_EEG_sub_1(p,:,t)=reshape(squeeze(X(p,10:18,1:9,t)),[],1);
        RDM_EEG_sub_2(p,:,t)=reshape(squeeze(X(p,19:27,1:9,t)),[],1);
        RDM_EEG_sub_3(p,:,t)=reshape(squeeze(X(p,19:27,10:18,t)),[],1);
    end
end

% average across participants
M_RDM_EEG=squeeze(mean(RDM_EEG,1));
M_RDM_EEG_sub_1=squeeze(mean(RDM_EEG_sub_1,1));
M_RDM_EEG_sub_2=squeeze(mean(RDM_EEG_sub_2,1));
M_RDM_EEG_sub_3=squeeze(mean(RDM_EEG_sub_3,1));

% RSA with the models
RSA_EEG_animacy=corr(M_RDM_EEG,squareform(model_animacy)',"type","Spearman")';
RSA_EEG_appearance=corr(M_RDM_EEG,squareform(model_appearance)',"type","Spearman")';

%%% random sampling and plot
RSApp_animal=[];
RSApp_appearance=[];
for pp=1:number_perm_model
    idx_subject=randsample(size(RDM_EEG,1),size(RDM_EEG,1),1);
    RDM_EEGpp=squeeze(mean(RDM_EEG(idx_subject,:,:),1));
    RSApp_animal(pp,:)=corr(RDM_EEGpp,squareform(model_animacy)',"type","Spearman")';
    RSApp_appearance(pp,:)=corr(RDM_EEGpp,squareform(model_appearance)',"type","Spearman")';
    pp;
end

% plot
close all
x_plot=figure(1); hold on
co1=brewermap(9,'Blues');
co2=brewermap(9,'Reds');
co = [co1(5,:);co2(5,:)];
x=-200:4:500;
error_animal=prctile(RSApp_animal,[5 95]); error_animal=error_animal([2,1],:);error_animal=error_animal-mean(RSApp_animal);
error_appearance=prctile(RSApp_appearance,[5 95]);error_appearance=error_appearance([2,1],:); error_appearance=error_appearance-mean(RSApp_appearance);
error_diff=prctile(RSApp_animal-RSApp_appearance,[2.5 97.5]);

yline(0,LineWidth=0.8,LineStyle=":")
shadedErrorBar(x, RSA_EEG_animacy,abs(error_animal),'lineprops',{'color',co(1,:),'linewidth',2})
shadedErrorBar(x, RSA_EEG_appearance,abs(error_appearance),'lineprops',{'color',co(2,:),'linewidth',2})

t_animacy=find(error_animal(2,:)+RSA_EEG_animacy>0);
plot(x(t_animacy),ones(1,length(t_animacy))*(-0.16),'color',co(1,:),marker=".",MarkerSize=13,LineStyle="none")
t_appearance=find(error_appearance(2,:)+RSA_EEG_appearance>0);
plot(x(t_appearance),ones(1,length(t_appearance))*(-0.20),'color',co(2,:),marker=".",MarkerSize=13,LineStyle="none")
t_diff=find(error_diff(2,:)<0|error_diff(1,:)>0);
plot(x(t_diff),ones(1,length(t_diff))*(-0.24),'color',[0.7,0.7,0.7],marker=".",MarkerSize=13,LineStyle="none")

set(gca,'TickDir','out');
[~, hobj, ~, ~] =legend({'','Animal model','Lookalike model'},'Location','northwest');legend boxoff
hl = findobj(hobj,'type','line');
set(hl,'LineWidth',4);
ht = findobj(hobj,'type','text');
set(ht,'FontSize',16);
set(gca, 'FontSize',16,'YTick',[0 0.5 1])
xlabel('EEG time (ms)')
ylabel('Spearman correlation')
xlim([0 500])
ylim([-0.28 1])
x_plot.Units = 'inches';
x_plot.OuterPosition = [1.5 1.5 25*0.4 10*0.6];
exportgraphics(x_plot,[figure_dir 'RSA_EEG_exp2.tif'],'resolution',600)

% RSA with mvt
for p=1:size(RDM_EEG,1)
    p
    corr_EEG_mvt_A(p,:,:)=corr(M_RDM_traj,squeeze(RDM_EEG(p,:,:)),"type","Spearman");
    corr_EEG_mvt_B(p,:,:)=partialcorr(M_RDM_traj,squeeze(RDM_EEG(p,:,:)),squareform(model_animacy)',"type","Spearman");

    corr_EEG_mvt_CB_A(p,:,:)=partialcorr(M_RDM_traj,squeeze(RDM_EEG(p,:,:)),[squareform(RDM_behav)',squareform(RDM_exec)'],"type","Spearman");
    corr_EEG_mvt_CB_B(p,:,:)=partialcorr(M_RDM_traj,squeeze(RDM_EEG(p,:,:)),[squareform(RDM_behav)',squareform(RDM_exec)',squareform(model_animacy)'],"type","Spearman");

    corr_EEG_mvt_sub1(p,:,:)=corr(M_RDM_traj_sub_1,squeeze(RDM_EEG_sub_1(p,:,:)),"type","Spearman");
    corr_EEG_mvt_sub2(p,:,:)=corr(M_RDM_traj_sub_2,squeeze(RDM_EEG_sub_2(p,:,:)),"type","Spearman");
    corr_EEG_mvt_sub3(p,:,:)=corr(M_RDM_traj_sub_3,squeeze(RDM_EEG_sub_3(p,:,:)),"type","Spearman");

    corr_EEG_mvt_CB_sub1(p,:,:)=partialcorr(M_RDM_traj_sub_1,squeeze(RDM_EEG_sub_1(p,:,:)),[reshape(RDM_behav(10:18,1:9),[],1),reshape(RDM_exec(10:18,1:9),[],1)],"type","Spearman");
    corr_EEG_mvt_CB_sub2(p,:,:)=partialcorr(M_RDM_traj_sub_2,squeeze(RDM_EEG_sub_2(p,:,:)),[reshape(RDM_behav(19:27,1:9),[],1),reshape(RDM_exec(19:27,1:9),[],1)],"type","Spearman");
    corr_EEG_mvt_CB_sub3(p,:,:)=partialcorr(M_RDM_traj_sub_3,squeeze(RDM_EEG_sub_3(p,:,:)),[reshape(RDM_behav(19:27,10:18),[],1),reshape(RDM_exec(19:27,10:18),[],1)],"type","Spearman");
end

%%%% RSA with fMRI
% reorder stimuli and average the two tasks
ROI_EVC=(OSF.brainData.brain_sim_singSubj_task_1{1, 1}(idx_stim,idx_stim,:)+OSF.brainData.brain_sim_singSubj_task_2{1, 1}(idx_stim,idx_stim,:))/2;
ROI_postVTC=(OSF.brainData.brain_sim_singSubj_task_1{1, 2}(idx_stim,idx_stim,:)+OSF.brainData.brain_sim_singSubj_task_2{1, 2}(idx_stim,idx_stim,:))/2;
ROI_antVTC=(OSF.brainData.brain_sim_singSubj_task_1{1, 3}(idx_stim,idx_stim,:)+OSF.brainData.brain_sim_singSubj_task_2{1, 3}(idx_stim,idx_stim,:))/2;

ROI_EVC=1-ROI_EVC;
ROI_postVTC=1-ROI_postVTC;
ROI_antVTC=1-ROI_antVTC;

for ii=1:27
    ROI_EVC(ii,ii,:)=0;
    ROI_postVTC(ii,ii,:)=0;
    ROI_antVTC(ii,ii,:)=0;
end

% average across participants
M_ROI_EVC=mean(ROI_EVC,3);
M_ROI_postVTC=mean(ROI_postVTC,3);
M_ROI_antVTC=mean(ROI_antVTC,3);


%RDMs = {model_animacy, model_appearance}; MV
RDMs = {M_ROI_EVC, M_ROI_postVTC,M_ROI_antVTC};
filenames = {'RDM_EVC_exp2.tif' 'RDM_postVTC_exp2.tif','RDM_antVTC_exp2.tif'};
for i=1:3
    close all
    x_plot=figure(1);
    rdm=RDMs{i};
    loweridx = find(tril(ones(size(rdm,1)),-1));
    rdm2 = normalize(rdm(loweridx),'range');
    imagesc(squareform(rdm2))
    set(gca,'TickDir', 'none')
    axis equal
    ylim([0.5 27.5])
    xlim([0.5 27.5])
    colormap magma
    x_plot.Units = 'inches';
    x_plot.OuterPosition = [1.5 1.5 5 5];
    exportgraphics(x_plot,[figure_dir filenames{i}],'resolution',300)
end

% Commonalities
for p = 1:size(RDM_EEG,1)
    p
    r_all(p,:,:)=partialcorr(M_RDM_traj,squeeze(RDM_EEG(p,:,:)),[squareform(M_ROI_EVC)',squareform(M_ROI_postVTC)',squareform(M_ROI_antVTC)',squareform(model_animacy)'],"type","Spearman");
    r_EVC(p,:,:)=partialcorr(M_RDM_traj,squeeze(RDM_EEG(p,:,:)),[squareform(M_ROI_postVTC)',squareform(M_ROI_antVTC)',squareform(model_animacy)'],"type","Spearman");
    r_postVTC(p,:,:)=partialcorr(M_RDM_traj,squeeze(RDM_EEG(p,:,:)),[squareform(M_ROI_EVC)',squareform(M_ROI_antVTC)',squareform(model_animacy)'],"type","Spearman");
    r_antVTC(p,:,:)=partialcorr(M_RDM_traj,squeeze(RDM_EEG(p,:,:)),[squareform(M_ROI_EVC)',squareform(M_ROI_postVTC)',squareform(model_animacy)'],"type","Spearman");
end


comm_EVC = r_EVC.^2 - r_all.^2;
comm_postVTC = r_postVTC.^2 - r_all.^2;
comm_antVTC = r_antVTC.^2 - r_all.^2;


nbr_perm=number_perm_cluster;
cluster_threshold=0.05;
cluster_measure='sum';

data = {corr_EEG_mvt_A, corr_EEG_mvt_CB_A, corr_EEG_mvt_B, corr_EEG_mvt_CB_B};
filenames = {'fusion_A_exp2.tif', 'fusion_CB_A_exp2.tif', 'fusion_B_exp2.tif', 'fusion_CB_B_exp2.tif'};
for i=1:4
    close all
    time_time_plot(data{i},2,nbr_perm,cluster_threshold,cluster_measure,1,0,0,[0 .08],[figure_dir filenames{i}],1);
end

data = {corr_EEG_mvt_sub1, corr_EEG_mvt_CB_sub1, corr_EEG_mvt_sub2,...
    corr_EEG_mvt_CB_sub2, corr_EEG_mvt_sub3, corr_EEG_mvt_CB_sub3};
filenames = {'fusion_sub1_exp2.tif', 'fusion_CB_sub1_exp2.tif', 'fusion_sub2_exp2.tif',...
    'fusion_CB_sub2_exp2.tif','fusion_sub3_exp2.tif','fusion_CB_sub3_exp2.tif'};
for i=1:6
    time_time_plot(data{i},2,nbr_perm,cluster_threshold,cluster_measure,1,0,1,[0 .15],[figure_dir filenames{i}],1);
end

data = {comm_EVC, comm_postVTC, comm_antVTC};
filenames = {'comm_EVC_exp2.tif', 'comm_postVTC_exp2.tif', 'comm_antVTC_exp2.tif'};
for i=1:3 %MV
    close all
    time_time_plot(data{i},2,nbr_perm,cluster_threshold,cluster_measure,0,0,0,[0 .004],[figure_dir filenames{i}],2);
end
