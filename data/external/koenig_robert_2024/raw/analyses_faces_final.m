%%%%%
clearvars

figure_dir = './Rstudio_traj/figures/';

load("trajectories-faces.mat")

number_perm_model=10000;
number_perm_cluster=10000;

% remove bad subjects and trials
TT=TT(TT.include==1,:);
%TT=TT(TT.include==1 & TT.correct==1,:); to run on correct trials only

%renumber subjects
[~,~,TT.subject] = unique(TT.subject);



% compute execution time RDM
threshold_x = 270;
threshold_y = 380;
[~,TT.execution_time]=max(abs(TT.xposi)>threshold_x & abs(TT.yposi)>threshold_y,[],2);
TT.execution_time(TT.execution_time==1) = NaN;
R=[];
for p=1:max(TT.subject) %each separately to give the same weight
    for i = 1:96 % nbr of stimuli
        int=TT.execution_time(TT.subject==p & TT.sortedstimulusnr==i,:);
        R(p,i)=nanmean(int,1);
    end
end



RDM_exec=pdist(nanmean(R,1)');
RDM_exec=squareform(RDM_exec);
x_plot=figure(1);
rdm=RDM_exec;
loweridx = find(tril(ones(size(rdm,1)),-1));
rdm2 = normalize(rdm(loweridx),'range');
imagesc(squareform(rdm2))
set(gca,'TickDir', 'none')
axis equal
ylim([0.5 96.5])
xlim([0.5 96.5])
colormap magma
x_plot.Units = 'inches';
x_plot.OuterPosition = [1.5 1.5 5 5];
exportgraphics(x_plot,[figure_dir 'RDM_exec_exp1.tif'],'resolution',300)


%%%% MEG
% load MEG,fMRI, and face ratings data from Wardle et al 2020
load('SourceData/Figure2c.mat')
load('SourceData/Figure5b.mat')
load('SourceData/Figure6a.mat')
load('SourceData/Figure6e.mat','timevect')
mt = timevect;

% Face ratings RDM
RDM_behav=turkRDM;
for i=1:96
    RDM_behav(i,i)=0;
end

close all
x_plot=figure(1);
rdm=RDM_behav;
loweridx = find(tril(ones(size(rdm,1)),-1));
rdm2 = normalize(rdm(loweridx),'range');
imagesc(squareform(rdm2))
set(gca, 'TickDir', 'none')
axis equal
ylim([0.5 96.5])
xlim([0.5 96.5])
colormap magma
x_plot.Units = 'inches';
x_plot.OuterPosition = [1.5 1.5 5 5];
exportgraphics(x_plot,[figure_dir 'RDM_behav_exp1.tif'],'resolution',300)

% compute mean trajectory for each stimulus and participant
M=[];
for p=1:max(TT.subject) %each separately to give the same weight
    for i = 1:96 % nbr of stimuli
        int=TT.xposi(TT.subject==p & TT.sortedstimulusnr==i ,:);
        M(p,i,:)=nanmean(int,1);
    end
end

My=[];
for p=1:max(TT.subject) %each separately to give the same weight
    for i = 1:96 % nbr of stimuli
        int=TT.yposi(TT.subject==p & TT.sortedstimulusnr==i ,:);
        My(p,i,:)=nanmean(int,1);
    end
end

% average across participants and plot
MM=M; MMy=My;
M=squeeze(nanmean(M,1));
My=squeeze(nanmean(My,1));

%%% individuals trials example
idx_part=[3 53 153];
co = repelem(brewermap(3,'Set2'),1,1); %96 objects
co(:,4)=0.8;

for ee=1:3
    int_TT=TT(TT.subject==idx_part(ee),:);
    close all; x_plot=figure(1);
    hold on
    for i=1:108 %size(int_TT,1)% plot 108 trials (matches the 108 trials of exp2)
        if int_TT.sortedstimulusnr(i)<33
            plot(int_TT.xposi(i,:),int_TT.yposi(i,:),'color',co(1,:),'linewidth',1.2)
        elseif int_TT.sortedstimulusnr(i)<65
            plot(int_TT.xposi(i,:),int_TT.yposi(i,:),'color',co(2,:),'linewidth',1.2)
        elseif int_TT.sortedstimulusnr(i)<97
            plot(int_TT.xposi(i,:),int_TT.yposi(i,:),'color',co(3,:),'linewidth',1.2)
        end
    end
    ylim([0 600])
    xlim([-600 600])
    axis equal
    exportgraphics(x_plot,[figure_dir 'Mean_traj_exp1_' num2str(ee) '.tif'],'resolution',300)
end

%%%%% Mean trajectories
co = repelem(brewermap(3,'Set2'),1,1); %96 objects
co(:,4)=0.8;
close all; x_plot=figure(1);

plot(zeros(511,1),-5:505,'k:', LineWidth=1)
hold on
x1=plot(M(1:32,:)',My(1:32,:)','color',co(1,:),'linewidth',0.8)
x2=plot(M(33:64,:)',My(33:64,:)','color',co(2,:),'linewidth',0.8)
x3=plot(M(65:96,:)',My(65:96,:)','color',co(3,:),'linewidth',0.8)

hleglines = [x1(1) x2(1) x3(1)];
ylim([0 500])
xlim([-330 330])
axis equal
xlabel('Horizontal position (pixels)')
ylabel('Vertical position (pixels)')
[~, hobj, ~, ~] =legend(hleglines,{'Face','Lookalike','Object'},'Location','southwest');legend boxoff
hl = findobj(hobj,'type','line');
set(hl,'LineWidth',4);
ht = findobj(hobj,'type','text');
set(ht,'FontSize',14);
set(gca, 'FontSize',14,'XTick',[-300 0 300],'YTick',[0 250 500])

x_plot.Units = 'inches';
x_plot.OuterPosition = [1.5 1.5 5 5];
exportgraphics(x_plot,[figure_dir 'Mean_traj_exp1.tif'],'resolution',300)

%%%%% figure horizontal position
co = repelem(brewermap(3,'Set2'),1,1); %96 objects
co(:,4)=0.8;
close all; x_plot=figure(1);
plot(0:800,zeros(801,1),'k:', LineWidth=1)
hold on
x1=plot(0:800,M(1:32,:),'color',co(1,:),'linewidth',0.8);
x2=plot(0:800,M(33:64,:),'color',co(2,:),'linewidth',0.8);
x3=plot(0:800,M(65:96,:),'color',co(3,:),'linewidth',0.8);

hleglines = [x1(1) x2(1) x3(1)];
xlim([0 800])
ylim([-330 330])
ylabel('Horizontal position (pixels)')
xlabel('Movement time (ms)')
[~, hobj, ~, ~] =legend(hleglines,{'Face','Lookalike','Object'},'Location','southwest');legend boxoff
hl = findobj(hobj,'type','line');
set(hl,'LineWidth',4);
ht = findobj(hobj,'type','text');
set(ht,'FontSize',14);
set(gca, 'YDir','reverse','FontSize',14,'YTick',[-300 0 300])

x_plot.Units = 'inches';
x_plot.OuterPosition = [1.5 1.5 6 5];
exportgraphics(x_plot,[figure_dir 'Mean_xpos_exp1.tif'],'resolution',300)

% compute RDMs trajectories
for t=1:size(M,2)
    M_RDM_traj(:,t)=pdist(M(:,t));
    int=squareform(pdist(M(:,t)));
    M_RDM_traj_sub_1(:,t)=reshape(int(33:64,1:32),[],1);
    M_RDM_traj_sub_2(:,t)=reshape(int(65:96,1:32),[],1);
    M_RDM_traj_sub_3(:,t)=reshape(int(65:96,33:64),[],1);
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

    set(gca, 'YDir','reverse','TickDir', 'none')
    axis equal
    ylim([1 96])
    xlim([1 96])
    colormap magma
    x_plot.Units = 'inches';
    x_plot.OuterPosition = [1.5 1.5 5 5];
    exportgraphics(x_plot,[figure_dir 'RDM_traj_exp1_' num2str(t-1) '.tif'],'resolution',300)
end

close all
x_plot=figure(1);
co = repelem(brewermap(3,'Set2'),32,1); %96 objects
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
    exportgraphics(x_plot,[figure_dir 'MDS_traj_exp1_' num2str(t-1) '.tif'],'resolution',600)
end

%%% export legend MDS
close all
x_plot=figure(1);
co = repelem(brewermap(3,'Set2'),1,1); %96 objects
scatter([1 1 1],[1 1.5 2],290,co,'filled','MarkerFaceAlpha',.8,'MarkerEdgeAlpha',.8)
ax=gca;
ax.XColor='w';
ax.YColor='w';
x_plot.Units = 'inches';
x_plot.OuterPosition = [0 0 1 3];
exportgraphics(x_plot,[figure_dir 'MDS_legend.tif'],'resolution',600)

%%%% RSA with the models
model_face=ones(96,96);
model_face(1:32,1:32)=0;
model_face(33:96,33:96)=0;

model_appearance=ones(96,96);
model_appearance(1:64,1:64)=0;
model_appearance(65:96,65:96)=0;

models = {model_face, model_appearance};
filenames = {'RDM_model_face_exp1.tif', 'RDM_model_lookalike_exp1.tif'};
for i=1:2
    x_plot=figure(1);
    imagesc(models{i});
    set(gca, 'TickDir', 'none')
    axis equal
    ylim([0.5 96.5])
    xlim([0.5 96.5])
    colormap magma
    x_plot.Units = 'inches';
    x_plot.OuterPosition = [1.5 1.5 5 5];
    exportgraphics(x_plot,[figure_dir,filenames{i}],'resolution',300)
end

RSA_face=corr(M_RDM_traj,squareform(model_face)',"type","Spearman")';
RSA_appearance=corr(M_RDM_traj,squareform(model_appearance)',"type","Spearman")';

RSApp_face=[];
RSApp_appearance=[];

for pp=1:number_perm_model
    idx_subject=randsample(size(MM,1),size(MM,1),1);
    Mpp=squeeze(nanmean(MM(idx_subject,:,:),1));
    Mpp_RDM_traj=[];
    for t=1:size(Mpp,2)
        Mpp_RDM_traj(:,t)=pdist(Mpp(:,t));
    end
    RSApp_face(pp,:)=corr(Mpp_RDM_traj,squareform(model_face)',"type","Spearman")';
    RSApp_appearance(pp,:)=corr(Mpp_RDM_traj,squareform(model_appearance)',"type","Spearman")';
    pp
end

% plot
error_face=prctile(RSApp_face,[5 95]); error_face=error_face([2,1],:);error_face=error_face-mean(RSApp_face);
error_appearance=prctile(RSApp_appearance,[5 95]);error_appearance=error_appearance([2,1],:); error_appearance=error_appearance-mean(RSApp_appearance);
error_diff=prctile(RSApp_face-RSApp_appearance,[2.5 97.5]);
co1=brewermap(9,'Blues');
co2=brewermap(9,'Reds');
co = [co1(5,:);co2(5,:)];
x=0:800;

close all
x_plot=figure(1); hold on
yline(0,LineWidth=0.8,LineStyle=":")
shadedErrorBar(x, RSA_face,abs(error_face),'lineprops',{'color',co(1,:),'linewidth',2})
shadedErrorBar(x, RSA_appearance,abs(error_appearance),'lineprops',{'color',co(2,:),'linewidth',2})

t_face=find(error_face(2,:)+RSA_face>0);
plot(t_face,ones(1,length(t_face))*(-0.16),'color',co(1,:),marker=".",MarkerSize=13,LineStyle="none")
t_appearance=find(error_appearance(2,:)+RSA_appearance>0);
plot(t_appearance,ones(1,length(t_appearance))*(-0.20),'color',co(2,:),marker=".",MarkerSize=13,LineStyle="none")
t_diff=find(error_diff(2,:)<0|error_diff(1,:)>0);
plot(t_diff,ones(1,length(t_diff))*(-0.24),'color',[0.7,0.7,0.7],marker=".",MarkerSize=13,LineStyle="none")

set(gca,'TickDir','out');
[~, hobj, ~, ~] =legend({'','Face model','Lookalike model'},'Location','northwest');legend boxoff
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
exportgraphics(x_plot,[figure_dir 'RSA_traj_exp1.tif'],'resolution',600)

%%%% RSA with MEG
RDM_MEG=[];RDM_MEG_sub_1=[];RDM_MEG_sub_2=[];RDM_MEG_sub_3=[];
for p=1:size(megRDM,3)
    for t=1:221
        X=squareform(megRDM(:,t,p));
        RDM_MEG(p,:,t)=squareform(X);
        RDM_MEG_sub_1(p,:,t)=reshape(X(33:64,1:32),[],1);
        RDM_MEG_sub_2(p,:,t)=reshape(X(65:96,1:32),[],1);
        RDM_MEG_sub_3(p,:,t)=reshape(X(65:96,33:64),[],1);
    end
end
M_RDM_MEG=squeeze(mean(RDM_MEG,1));
M_RDM_MEG_sub_1=squeeze(mean(RDM_MEG_sub_1,1));
M_RDM_MEG_sub_2=squeeze(mean(RDM_MEG_sub_2,1));
M_RDM_MEG_sub_3=squeeze(mean(RDM_MEG_sub_3,1));

% visualisation over time
mt=-100:5:1000;
N=1;
for t=70:90:500 %%%% find closest time point
    idx_int=find(abs(mt-t)==min(abs(mt-t)));
    idx_time(N)=idx_int(1);
    N=N+1;
end

%%
close all
x_plot=figure(1);
for t=1:length(idx_time)
    hold on
    rdm = squareform(squeeze(M_RDM_MEG(:,idx_time(t))));
    loweridx = find(tril(ones(size(rdm,1)),-1));
    rdm2 = normalize(rdm(loweridx),'range');
    imagesc(squareform(rdm2))
    set(gca, 'YDir','reverse','TickDir', 'none')
    axis equal
    ylim([0.5 96.5])
    xlim([0.5 96.5])
    colormap magma
    x_plot.Units = 'inches';
    x_plot.OuterPosition = [1.5 1.5 5 5];
    exportgraphics(x_plot,[figure_dir 'RDM_MEG_exp1_' num2str(70+(t-1)*90) '.tif'],'resolution',300)
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
co = repelem(brewermap(3,'Set2'),32,1); %96 objects
mm=50;
for t=1:length(idx_time)
    rdm = squareform(squeeze(M_RDM_MEG(:,idx_time(t))));
    rng(1)
    Y = mdscale(rdm,2,'Criterion','stress');
    scatter(Y(:,1),Y(:,2),140,co,'filled','MarkerFaceAlpha',.8,'MarkerEdgeAlpha',.8)
    ax=gca;
    ax.XColor='w';
    ax.YColor='w';
    x_plot.Units = 'inches';
    x_plot.OuterPosition = [1.5 1.5 5 5];
    exportgraphics(x_plot,[figure_dir 'MDS_MEG_exp1_' num2str(70+(t-1)*45) '.tif'],'resolution',600)
end

%%% RSA with the models
RSA_MEG_face=corr(M_RDM_MEG,squareform(model_face)',"type","Spearman")';
RSA_MEG_appearance=corr(M_RDM_MEG,squareform(model_appearance)',"type","Spearman")';



















%%% random sampling and plot
RSApp_face=[];
RSApp_appearance=[];
for pp=1:number_perm_model
    idx_subject=randsample(size(RDM_MEG,1),size(RDM_MEG,1),1);
    RDM_MEGpp=squeeze(mean(RDM_MEG(idx_subject,:,:),1));
    RSApp_face(pp,:)=corr(RDM_MEGpp,squareform(model_face)',"type","Spearman")';
    RSApp_appearance(pp,:)=corr(RDM_MEGpp,squareform(model_appearance)',"type","Spearman")';
    pp
end

% plot
close all
x_plot=figure(1); hold on
co1=brewermap(9,'Blues');
co2=brewermap(9,'Reds');
co = [co1(5,:);co2(5,:)];
x=-100:5:1000;
error_face=prctile(RSApp_face,[5 95]); error_face=error_face([2,1],:);error_face=error_face-mean(RSApp_face);
error_appearance=prctile(RSApp_appearance,[5 95]);error_appearance=error_appearance([2,1],:); error_appearance=error_appearance-mean(RSApp_appearance);
error_diff=prctile(RSApp_face-RSApp_appearance,[2.5 97.5]);

yline(0,LineWidth=0.8,LineStyle=":")
shadedErrorBar(x, RSA_MEG_face,abs(error_face),'lineprops',{'color',co(1,:),'linewidth',2})
shadedErrorBar(x, RSA_MEG_appearance,abs(error_appearance),'lineprops',{'color',co(2,:),'linewidth',2})

t_face=find(error_face(2,:)+RSA_MEG_face>0);
plot(x(t_face),ones(1,length(t_face))*(-0.16),'color',co(1,:),marker=".",MarkerSize=13,LineStyle="none")
t_appearance=find(error_appearance(2,:)+RSA_MEG_appearance>0);
plot(x(t_appearance),ones(1,length(t_appearance))*(-0.20),'color',co(2,:),marker=".",MarkerSize=13,LineStyle="none")
t_diff=find(error_diff(2,:)<0|error_diff(1,:)>0);
plot(x(t_diff),ones(1,length(t_diff))*(-0.24),'color',[0.7,0.7,0.7],marker=".",MarkerSize=13,LineStyle="none")

set(gca,'TickDir','out');
[~, hobj, ~, ~] =legend({'','Face model','Lookalike model'},'Location','northwest');legend boxoff
hl = findobj(hobj,'type','line');
set(hl,'LineWidth',4);
ht = findobj(hobj,'type','text');
set(ht,'FontSize',16);
set(gca, 'FontSize',16,'YTick',[0 0.5 1])
xlabel('MEG time (ms)')
ylabel('Spearman correlation')
xlim([0 500])
ylim([-0.28 1])
x_plot.Units = 'inches';
x_plot.OuterPosition = [1.5 1.5 25*0.4 10*0.6];
exportgraphics(x_plot,[figure_dir 'RSA_MEG_exp1.tif'],'resolution',600)

% RSA with mvt
for p=1:size(RDM_MEG,1)
    p
    corr_MEG_mvt_A(p,:,:)=corr(M_RDM_traj,squeeze(RDM_MEG(p,:,:)),"type","Spearman");
    corr_MEG_mvt_B(p,:,:)=partialcorr(M_RDM_traj,squeeze(RDM_MEG(p,:,:)),squareform(model_face)',"type","Spearman");

    corr_MEG_mvt_CB_A(p,:,:)=partialcorr(M_RDM_traj,squeeze(RDM_MEG(p,:,:)),[squareform(RDM_behav)',squareform(RDM_exec)'],"type","Spearman");
    corr_MEG_mvt_CB_B(p,:,:)=partialcorr(M_RDM_traj,squeeze(RDM_MEG(p,:,:)),[squareform(RDM_behav)',squareform(RDM_exec)',squareform(model_face)'],"type","Spearman");

    corr_MEG_mvt_sub1(p,:,:)=corr(M_RDM_traj_sub_1,squeeze(RDM_MEG_sub_1(p,:,:)),"type","Spearman");
    corr_MEG_mvt_sub2(p,:,:)=corr(M_RDM_traj_sub_2,squeeze(RDM_MEG_sub_2(p,:,:)),"type","Spearman");
    corr_MEG_mvt_sub3(p,:,:)=corr(M_RDM_traj_sub_3,squeeze(RDM_MEG_sub_3(p,:,:)),"type","Spearman");

    corr_MEG_mvt_CB_sub1(p,:,:)=partialcorr(M_RDM_traj_sub_1,squeeze(RDM_MEG_sub_1(p,:,:)),[reshape(RDM_behav(33:64,1:32),[],1),reshape(RDM_exec(33:64,1:32),[],1)],"type","Spearman");
    corr_MEG_mvt_CB_sub2(p,:,:)=partialcorr(M_RDM_traj_sub_2,squeeze(RDM_MEG_sub_2(p,:,:)),[reshape(RDM_behav(65:96,1:32),[],1),reshape(RDM_exec(65:96,1:32),[],1)],"type","Spearman");
    corr_MEG_mvt_CB_sub3(p,:,:)=partialcorr(M_RDM_traj_sub_3,squeeze(RDM_MEG_sub_3(p,:,:)),[reshape(RDM_behav(65:96,33:64),[],1),reshape(RDM_exec(65:96,33:64),[],1)],"type","Spearman");
end

%%%% RSA with fMRI
load('SourceData/Figure2c.mat')
ROI_ffa=rdm.ffa;
ROI_ofa=rdm.ofa;
ROI_ppa=rdm.ppa;
ROI_lo=rdm.lo;

for i=1:96
    ROI_ffa(i,i,:)=0;
    ROI_ofa(i,i,:)=0;
    ROI_ppa(i,i,:)=0;
    ROI_lo(i,i,:)=0;
end



% average across participants
M_ROI_ffa=mean(ROI_ffa,3);
M_ROI_ofa=mean(ROI_ofa,3);
M_ROI_ppa=mean(ROI_ppa,3);
M_ROI_lo=mean(ROI_lo,3);

RDMs = {M_ROI_ffa, M_ROI_ofa,M_ROI_ppa,M_ROI_lo};
filenames = {'RDM_ffa_exp1.tif' 'RDM_ofa_exp1.tif','RDM_ppa_exp1.tif','RDM_lo_exp1.tif'};
for i=1:4
    close all
    x_plot=figure(1);
    rdm=RDMs{i};
    loweridx = find(tril(ones(size(rdm,1)),-1));
    rdm2 = normalize(rdm(loweridx),'range');
    imagesc(squareform(rdm2))
    set(gca,'TickDir', 'none')
    axis equal
    ylim([0.5 96.5])
    xlim([0.5 96.5])
    colormap magma
    x_plot.Units = 'inches';
    x_plot.OuterPosition = [1.5 1.5 5 5];
    exportgraphics(x_plot,[figure_dir filenames{i}],'resolution',300)
end

% Commonalities
for p = 1:size(RDM_MEG,1)
    p
    r_all(p,:,:)=partialcorr(M_RDM_traj,squeeze(RDM_MEG(p,:,:)),[squareform(M_ROI_ffa)',squareform(M_ROI_ofa)',squareform(M_ROI_ppa)',squareform(M_ROI_lo)',squareform(model_face)'],"type","Spearman");
    r_ffa(p,:,:)=partialcorr(M_RDM_traj,squeeze(RDM_MEG(p,:,:)),[squareform(M_ROI_ofa)',squareform(M_ROI_ppa)',squareform(M_ROI_lo)',squareform(model_face)'],"type","Spearman");
    r_ofa(p,:,:)=partialcorr(M_RDM_traj,squeeze(RDM_MEG(p,:,:)),[squareform(M_ROI_ffa)',squareform(M_ROI_ppa)',squareform(M_ROI_lo)',squareform(model_face)'],"type","Spearman");
    r_ppa(p,:,:)=partialcorr(M_RDM_traj,squeeze(RDM_MEG(p,:,:)),[squareform(M_ROI_ffa)',squareform(M_ROI_ofa)',squareform(M_ROI_lo)',squareform(model_face)'],"type","Spearman");
    r_lo(p,:,:)=partialcorr(M_RDM_traj,squeeze(RDM_MEG(p,:,:)),[squareform(M_ROI_ffa)',squareform(M_ROI_ofa)',squareform(M_ROI_ppa)',squareform(model_face)'],"type","Spearman");
end

comm_ffa = r_ffa.^2 - r_all.^2;
comm_ofa = r_ofa.^2 - r_all.^2;
comm_ppa = r_ppa.^2 - r_all.^2;
comm_lo = r_lo.^2 - r_all.^2;

nbr_perm=number_perm_cluster;
cluster_threshold=0.05;
cluster_measure='sum';

data = {corr_MEG_mvt_A, corr_MEG_mvt_CB_A, corr_MEG_mvt_B, corr_MEG_mvt_CB_B};
filenames = {'fusion_A_exp1.tif', 'fusion_CB_A_exp1.tif', 'fusion_B_exp1.tif', 'fusion_CB_B_exp1.tif'};
for i=1:4
    close all
    time_time_plot(data{i},1,nbr_perm,cluster_threshold,cluster_measure,1,0,0,[0 .08],[figure_dir filenames{i}],1);
end

data = {corr_MEG_mvt_sub1, corr_MEG_mvt_CB_sub1, corr_MEG_mvt_sub2,...
    corr_MEG_mvt_CB_sub2, corr_MEG_mvt_sub3, corr_MEG_mvt_CB_sub3};
filenames = {'fusion_sub1_exp1.tif', 'fusion_CB_sub1_exp1.tif', 'fusion_sub2_exp1.tif',...
    'fusion_CB_sub2_exp1.tif','fusion_sub3_exp1.tif','fusion_CB_sub3_exp1.tif'};
for i=1:6
    time_time_plot(data{i},1,nbr_perm,cluster_threshold,cluster_measure,1,0,1,[0 .06],[figure_dir filenames{i}],1);
end

data = {comm_ffa, comm_ofa, comm_ppa, comm_lo};
filenames = {'comm_ffa_exp1.tif', 'comm_ofa_exp1.tif', 'comm_ppa_exp1.tif', 'comm_lo_exp1.tif'};
for i=1:4
    close all
    time_time_plot(data{i},1,nbr_perm,cluster_threshold,cluster_measure,0,0,0,[0 .001],[figure_dir filenames{i}],2);
end
