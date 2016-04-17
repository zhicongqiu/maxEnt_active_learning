function [ERROR SPARSITY NormalVsRare] = ...
		 active_learning(data_raw,data_p,data_GT,test_raw,test_p,test_GT,...
						 normal_class,rare_class,num_train,ITERATION,what_strategy,pval4R)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%By: Zhicong Qiu
%EECS
%The Penn State University
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%input: 
%data_raw: training and unlabeled sample in raw space
%
%data_p: training and unlabeled sample in p-value space
%
%data_GT: training and unlabeled sample labels
%
%test_raw: test sample in raw space
%
%test_p: test sample in p-value space
%
%test_GT: test sample labels
%
%normal_class: normal classes index
%
%rare_class: rare classes index
%
%num_train: 
%number of initial training samples. The normal subset is first used in GMM modelings and assigning p-values
%
%ITERATION: 
%number of AL iterations
%
%what_strategy: 
%0 for top-level uncertainty sampling 
%1 for for suspicious sampling, 
%2 for all-level uncertainty sampling
%3 for most likely unknown sampling (=0 if no labeled rare sample)
%-1 for random sampling
%
%pval4R: either use p-value for 2nd level (=1) or use raw for 2nd level (=0)
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%output:
%
%ERROR: 
%several error rate report as a function of AL. See definition for details
%
%SPARSITY: sparsity measure for both top and bottom level classifiers. See definition for details
%
%NormalVsRare: 
%normal vs rare catogery metrics, with the following fields:
%TD: true detections as a function of AL
%FA: false alarms as a function of AL
%discoverR: number of rare class discovered as a function of AL
%discoverN: number of normal class discovered as a function of AL
%ROC_AUC: roc auc as a function of AL
%label_stream: what data are forwarded to labeling
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%%


%make sure train and test has the same input dimension
if size(data_raw,2)~=size(test_raw,2)
   error('train and test dimension mismatch')
end
%make sure number of samples is the same as number of labels
if length(data_raw)~=length(data_GT)
   error('number of training samples should be equal to number of labels')
end
if length(test_raw)~=length(test_GT)
   error('number of test samples should be equal to number of labels')
end


N_O = size(data_raw,2);
N_P = size(data_p,2);
M = length(data_GT);
label = 2*ones(M,1); %2 is unlabeled, 1 is rare, 0 is normal
if num_train > M
	fprintf('num_train larger than training batch size? Reset to training batach size');
	num_train = M; %at most all labeled
end

%ground-truth transformation, 1st column indicates its Param index; 
%2nd column indicates if it is normal (=0)
data_GTT = transform_GT(data_GT,normal_class,rare_class);
test_GTT = transform_GT(test_GT,normal_class,rare_class);

%initialize active sets, indicating what classes have been discovered
active_set_normal = zeros(length(normal_class),1);
active_set_rare = zeros(length(rare_class),1);
num_normal_labeled = 0;
num_rare_labeled = 0;
for i=1:num_train
    if data_GTT(i,2)==0
        if active_set_normal(data_GTT(i,1))==0
            active_set_normal(data_GTT(i,1)) = 1;
        end
        label(i) = 0;
        num_normal_labeled = num_normal_labeled+1;
    elseif data_GTT(i,2)==1
        if active_set_rare(data_GTT(i,1))==0
            active_set_rare(data_GTT(i,1)) = 1;
        end
        label(i) = 1;
        num_rare_labeled = num_rare_labeled+1;
    end
end

%parameter initialization
ParamN = struct;
ParamR = struct;
ParamPR = struct;

%unbiasedly chosen initialization
for i=1:length(normal_class)    
    ParamN(i).beta = 1e-6*ones(1,N_O);
    ParamN(i).beta0 = 0;
end
for i=1:length(rare_class)   
    if pval4R==false
        ParamR(i).beta = 1e-6*ones(1,N_O);
    else
        ParamR(i).beta = 1e-6*ones(1,N_P);
    end
    ParamR(i).beta0 = 0;
end
ParamPR.alpha = 1e-6*ones(1,N_P);
ParamPR.alpha0 = 0;

SPARSITY = struct;
%initialize performance measure vectors
ERROR = struct;
%classification error
ERROR.error = zeros(ITERATION,1);
%false alarm rate
ERROR.FAR = zeros(ITERATION,1);
%false negative rate
ERROR.FNR = zeros(ITERATION,1);
%average error on the normal categories
ERROR.error_CN = zeros(ITERATION,1);
%average error on the rare categories
ERROR.error_CR = zeros(ITERATION,1);
%average classification error per-class
ERROR.error_avgC = zeros(ITERATION,1);
%average error on the unknown classes
ERROR.unknown_error_avgC = zeros(ITERATION,1);

%normal vs rare metrics
NormalVsRare = struct;
%true detections
NormalVsRare.TD = zeros(ITERATION,1);
%false positives
NormalVsRare.FA = zeros(ITERATION,1);
%Area under the Receiver Operating Characteristic 
NormalVsRare.ROC_AUC = zeros(ITERATION,1);
%fraction of rare classes discovered
NormalVsRare.discoverR = zeros(ITERATION,1);
%fraction of normal classes discovered
NormalVsRare.discoverN = zeros(ITERATION,1);
%which samples are forwarded to the oracle for labeling
NormalVsRare.label_stream = zeros(ITERATION,1);
%%

%begin AL 
for i=1:ITERATION
    %choose a_u through cross-validation, or using fixed value for faster
    %computation
    a_u = 0.5; %DoCV(data_raw,N_O,data_p,N_P,data_GTT,label,active_set_normal,active_set_rare,pval4R,true);
    fprintf('this is %d iterations, with a_u=%g...\n',i,a_u);
    fprintf('begin GD...\n');
    if pval4R==false %use raw features for 2nd level
        [ParamPR ParamN ParamR] = ...
        gradDes(data_raw,data_p,data_GTT,label,a_u,ParamPR,ParamN,ParamR,active_set_normal,active_set_rare,num_normal_labeled,num_rare_labeled,pval4R);
    else %use p-value features for 2nd level
        [ParamPR ParamN ParamR] = ...
        gradDes(data_p,data_p,data_GTT,label,a_u,ParamPR,ParamN,ParamR,active_set_normal,active_set_rare,num_normal_labeled,num_rare_labeled,pval4R);	
    end	
	
    %test PMF based on current parameters
    [test_PMFunknown test_PMFnormal test_PMFrare] = ...
	getTestPMF(test_raw,test_p,test_GTT,ParamPR,ParamN,ParamR,active_set_normal,active_set_rare,pval4R);
    
    %sparsity measure
    SPARSITY(i).ParamPR = ParamPR;
    SPARSITY(i).PR_mag = norm(ParamPR.alpha);
    SPARSITY(i).PR_sparse = sum(ParamPR.alpha==0)/length(ParamPR.alpha);    
    SPARSITY(i).ParamN = ParamN(active_set_normal==1);
    SPARSITY(i).ParamR = ParamR(active_set_rare==1);
    SPARSITY(i).active_set_normal = active_set_normal;
    SPARSITY(i).active_set_rare = active_set_rare;
    
    %test set performance measure
    NormalVsRare.ROC_AUC(i) = calculate_ROC(test_PMFunknown,test_GTT);
    %display ROC_AUC
    fprintf('test ROC_AUC is %g\n',NormalVsRare.ROC_AUC(i));
    
    %error rate on test set
    [ERROR.error(i), ERROR.FAR(i), ERROR.FNR(i), ERROR.error_CN(i), ...
	ERROR.error_CR(i), ERROR.error_avgC(i), ERROR.unknown_error_avgC(i)] = ...
	calculate_error(test_PMFunknown,test_PMFnormal,test_PMFrare,test_GTT,active_set_rare,0);
    fprintf('average class error rare is %g\n',ERROR.error_avgC(i));
    %active selection for labeling
    PMF_train = getF(data_p,ParamPR);
    PMF_rare = 0;
    PMF_normal = 0;
    tempR_sum = sum(active_set_rare);
    if tempR_sum>=1
        if pval4R==true
            PMF_rare = ...
			getF_multiclassRare(data_p,data_GTT,2*ones(size(data_p,1),1),ParamR,active_set_rare);
        else
            PMF_rare = ...
			getF_multiclassRare(data_raw,data_GTT,2*ones(size(data_raw,1),1),ParamR,active_set_rare);
        end
    end
    index_set = cumsum(ones(M,1));
    tempN_sum = sum(active_set_normal);
    if tempN_sum>=2
        PMF_normal = ...
		getF_multiclassNormal(data_raw,data_GTT,2*ones(size(data_raw,1),1),ParamN,active_set_normal);
    end
    if tempN_sum<2 && tempR_sum<1 && what_strategy==2
        what_strategy = 0;
    end
    sample2label = ...
	AL_strategy(PMF_train,PMF_rare,PMF_normal,index_set(label==2),active_set_rare,active_set_normal,what_strategy);
    NormalVsRare.label_stream(i) = sample2label;
    label(sample2label) = data_GTT(sample2label,2);
    if data_GTT(sample2label,2)==1 %rare sample
        num_rare_labeled = num_rare_labeled+1;
        if i==1
            fprintf('label a rare sample\n');
            NormalVsRare.TD(i) = 1;
            NormalVsRare.discoverR(i) = 1;
            active_set_rare(data_GTT(sample2label,1)) = 1;
        else
            NormalVsRare.TD(i) = NormalVsRare.TD(i-1)+1;
            NormalVsRare.FA(i) = NormalVsRare.FA(i-1);
             if active_set_rare(data_GTT(sample2label,1))==0
                 fprintf('label an unknown sample and discover an unknown class\n');
                 NormalVsRare.discoverR(i) = NormalVsRare.discoverR(i-1)+1;
                 active_set_rare(data_GTT(sample2label,1)) = 1;
             else
                 NormalVsRare.discoverR(i) = NormalVsRare.discoverR(i-1);
             end
             NormalVsRare.discoverN(i) = NormalVsRare.discoverN(i-1);
        end
    else %normal sample
        num_normal_labeled = num_normal_labeled+1;
        if i==1
            NormalVsRare.FA(i) = 1;
            if active_set_normal(data_GTT(sample2label,1))==0
                NormalVsRare.discoverN(i) = 1;
                active_set_normal(data_GTT(sample2label,1))=1;
            end
        else
            NormalVsRare.FA(i) = NormalVsRare.FA(i-1)+1;
            NormalVsRare.TD(i) = NormalVsRare.TD(i-1);
            if active_set_normal(data_GTT(sample2label,1))==0
                NormalVsRare.discoverN(i) = NormalVsRare.discoverN(i-1)+1;
                active_set_normal(data_GTT(sample2label,1))=1;
            else
                NormalVsRare.discoverN(i) = NormalVsRare.discoverN(i-1);
            end
            NormalVsRare.discoverR(i) = NormalVsRare.discoverR(i-1);
        end
    end    
end
end
    
    