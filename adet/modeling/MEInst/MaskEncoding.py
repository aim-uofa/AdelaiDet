# coding:utf-8
import torch
import torch.nn as nn

VALUE_MAX = 0.05
VALUE_MIN = 0.01


@torch.no_grad()
class PCAMaskEncoding(nn.Module):
    """
    To do the mask encoding of PCA.
        components_: (tensor), shape (n_components, n_features) if agnostic=True
                                else (n_samples, n_components, n_features)
        explained_variance_: Variance explained by each of the selected components.
                            (tensor), shape (n_components) if agnostic=True
                                        else (n_samples, n_components)
        mean_: (tensor), shape (n_features) if agnostic=True
                          else (n_samples, n_features)
        agnostic: (bool), whether class_agnostic or class_specific.
        whiten : (bool), optional
        When True (False by default) the ``components_`` vectors are divided
        by ``n_samples`` times ``components_`` to ensure uncorrelated outputs
        with unit component-wise variances.
        Whitening will remove some information from the transformed signal
        (the relative variance scales of the components) but can sometimes
        improve the predictive accuracy of the downstream estimators by
        making data respect some hard-wired assumptions.
        sigmoid: (bool) whether to apply inverse sigmoid before transform.
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.agnostic = cfg.MODEL.MEInst.AGNOSTIC
        self.whiten = cfg.MODEL.MEInst.WHITEN
        self.sigmoid = cfg.MODEL.MEInst.SIGMOID
        self.dim_mask = cfg.MODEL.MEInst.DIM_MASK
        self.mask_size = cfg.MODEL.MEInst.MASK_SIZE

        if self.agnostic:
            self.components = nn.Parameter(torch.zeros(self.dim_mask, self.mask_size**2), requires_grad=False)
            self.explained_variances = nn.Parameter(torch.zeros(self.dim_mask), requires_grad=False)
            self.means = nn.Parameter(torch.zeros(self.mask_size**2), requires_grad=False)
        else:
            raise NotImplementedError

    def inverse_sigmoid(self, x):
        """Apply the inverse sigmoid operation.
                y = -ln(1-x/x)
        """
        # In case of overflow
        value_random = VALUE_MAX * torch.rand_like(x)
        value_random = torch.where(value_random > VALUE_MIN, value_random, VALUE_MIN * torch.ones_like(x))
        x = torch.where(x > value_random, 1 - value_random, value_random)
        # inverse sigmoid
        y = -1 * torch.log((1 - x) / x)
        return y

    def encoder(self, X):
        """Apply dimensionality reduction to X.
        X is projected on the first principal components previously extracted
        from a training set.
        Parameters
        ----------
        X : Original features(tensor), shape (n_samples, n_features)
            New data, where n_samples is the number of samples
            and n_features is the number of features.

        Returns
        -------
        X_transformed : Transformed features(tensor), shape (n_samples, n_features)
        """
        assert X.shape[1] == self.mask_size**2, print("The original mask_size of input"
                                                      " should be equal to the supposed size.")

        if self.sigmoid:
            X = self.inverse_sigmoid(X)

        if self.agnostic:
            if self.means is not None:
                X_transformed = X - self.means
            X_transformed = torch.matmul(X_transformed, self.components.T)
            if self.whiten:
                X_transformed /= torch.sqrt(self.explained_variances)
        else:
            # TODO: The class-specific version has not implemented.
            raise NotImplementedError

        return X_transformed

    def decoder(self, X, is_train=False):
        """Transform data back to its original space.
        In other words, return an input X_original whose transform would be X.
        Parameters
        ----------
        X : Encoded features(tensor), shape (n_samples, n_components)
            New data, where n_samples is the number of samples
            and n_components is the number of components.

        Returns
        -------
        X_original original features(tensor), shape (n_samples, n_features)
        """
        assert X.shape[1] == self.dim_mask, print("The dim of transformed data "
                                                  "should be equal to the supposed dim.")

        if self.agnostic:
            if self.whiten:
                components_ = self.components * torch.sqrt(self.explained_variances.unsqueeze(1))
            X_transformed = torch.matmul(X, components_)
            if self.means is not None:
                X_transformed = X_transformed + self.means
        else:
            # TODO: The class-specific version has not implemented.
            raise NotImplementedError

        if is_train:
            pass
        else:
            if self.sigmoid:
                X_transformed = torch.sigmoid(X_transformed)
            else:
                X_transformed = torch.clamp(X_transformed, min=0.01, max=0.99)

        return X_transformed
