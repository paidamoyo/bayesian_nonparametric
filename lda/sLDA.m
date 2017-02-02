function model=sLDA(traindata,topics,em_max_iter,vbe_max_iter);
%%%%%%%%%%%%%%%%%%%%%%%%model parameters initialization%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
model.K=topics;
model.alpha = repmat(1/model.K, 1, model.K);
model.beta = repmat(1/traindata.dicwordnum, model.K, traindata.dicwordnum);
model.betas=repmat(1/model.K, traindata.dicwordnum, model.K);%to accumulate parameters
model.eta = repmat(0.0, model.K, 1);  
model.sigma = var(traindata.rate);
model.gammas = repmat(0.0, traindata.docnum, model.K);
for i=1:traindata.docnum
    gamma= model.alpha + repmat(traindata.doc(i).docwordnum/model.K, 1, model.K);
    model.gammas(i,:) =gamma;
end
model.phi=repmat(1/model.K, traindata.dicwordnum, model.K);

% additional two expectation values used for updating 'eta' and 'sigma'
E_A = repmat(0.0, traindata.docnum, model.K);
E_AA = repmat(0.0, model.K, model.K);

for iter=1:em_max_iter,
    corp.llhood = 0;
  
    % variational bayesian E-step
    % ===========================
    for i=1:traindata.docnum,
        % Some calculation in advance to save time
        doc=traindata.doc(i);
        
        npara_part1 = repmat(doc.rate/(doc.docwordnum*model.sigma)*model.eta' - model.eta'.*model.eta'...
        /(2*doc.docwordnum^2*model.sigma), length(doc.word_id), 1);
        
        eta_square = model.eta* model.eta';
        norm_part2 = 2 * doc.docwordnum^2 * model.sigma;

        for j=1:vbe_max_iter
            npara_part2 = 2*(repmat(sum(diag(doc.word)*model.phi(doc.word_id,:),1),length(doc.word_id),1)...
                - diag(doc.word)*model.phi(doc.word_id,:)) * eta_square / norm_part2;
            
            %update phi accroding to equation
            model.phi(doc.word_id,:) = model.beta(:,doc.word_id)'*diag(exp(psi(model.gammas(i,:))))...
                .*exp(npara_part1 - npara_part2);
            nm_const = sum(model.phi(doc.word_id,:), 2);
            model.phi(doc.word_id,:)= diag(1./nm_const)*model.phi(doc.word_id,:);
                        
            %update gamma according to equation
            gamma = model.alpha + doc.word*model.phi(doc.word_id,:);
            model.gammas(i,:) = gamma;
            
        end
        betas_sum = sum(diag(doc.word)*model.phi(doc.word_id,:), 1);
        E_A(i,:) = betas_sum./doc.docwordnum;  
        temp_E_AA = repmat(0.0, model.K, model.K);
        for k=1:length(doc.word),    
            for s=1:doc.word(k),
                temp_E_AA = temp_E_AA + model.phi(doc.word_id(k),:)'*(betas_sum - model.phi(doc.word_id(k),:))...
                    + diag(model.phi(doc.word_id(k), :)');
            end
        end
        E_AA = E_AA + temp_E_AA./ doc.docwordnum^2;
        
        model.betas(traindata.doc(i).word_id,:) = model.betas(traindata.doc(i).word_id,:)...
            + diag(traindata.doc(i).word)*model.phi(traindata.doc(i).word_id,:);
    end

    % variational bayesian M-step
    % ===========================
    % update beta 
    nm_const=sum(model.betas',2);
    model.beta = diag(1./nm_const) *model.betas';
 
    

    % update regression parameters eta, sigma.
    y = traindata.rate';
    E_AA_inv = inv(E_AA);
    model.eta = E_AA_inv*E_A'*y;
    model.sigma = (y'*y-y'*E_A*model.eta)/traindata.docnum;
%     
%     compute train data log-likelihood
%     =================================
%     [corp_llhood, perword_llhood]= slda_lik(traindata);
%     fprintf('Corpus log-likelihood = %f, ', corp_llhood);
%     fprintf('per-word log-likelihood = %f...\n',perword_llhood);
    
    % clear
    E_A(:,:) = 0.0;
    E_AA(:,:) = 0.0;
    model.betas(:,:) = 0;
end