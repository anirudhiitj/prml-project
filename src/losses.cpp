// ─────────────────────────────────────────────────────────────────────────────
// losses.cpp — SI-SNR + PIT + spectral PIT implementation
// ─────────────────────────────────────────────────────────────────────────────
#include "losses.h"

namespace losses {

torch::Tensor si_snr(torch::Tensor est, torch::Tensor tgt) {
    if (est.dim() == 1) est = est.unsqueeze(0);
    if (tgt.dim() == 1) tgt = tgt.unsqueeze(0);

    // Zero-mean
    est = est - est.mean(-1, true);
    tgt = tgt - tgt.mean(-1, true);

    // s_target = <est, tgt> / ||tgt||^2 * tgt
    auto dot    = (est * tgt).sum(-1, true);
    auto tgt_sq = (tgt * tgt).sum(-1, true);
    auto s_tgt  = dot * tgt / (tgt_sq + 1e-8);

    auto e_noise = est - s_tgt;
    return 10.0 * torch::log10(
        (s_tgt * s_tgt).sum(-1) / ((e_noise * e_noise).sum(-1) + 1e-8)
    );  // [B]
}

torch::Tensor pit_loss(torch::Tensor est, torch::Tensor tgt) {
    // est: [B, C, T], tgt: [B, C, T] with C=2
    // Perm 1: est[0]→tgt[0], est[1]→tgt[1]
    auto p1 = si_snr(est.select(1,0), tgt.select(1,0))
            + si_snr(est.select(1,1), tgt.select(1,1));
    // Perm 2: est[0]→tgt[1], est[1]→tgt[0]
    auto p2 = si_snr(est.select(1,0), tgt.select(1,1))
            + si_snr(est.select(1,1), tgt.select(1,0));
    return -torch::max(p1, p2).mean();
}

torch::Tensor spectral_pit_loss(torch::Tensor masks, torch::Tensor mix_mag,
                                 torch::Tensor tgt_mags) {
    // masks: [B, C, F, T], mix_mag: [B, 1, F, T], tgt_mags: [B, C, F, T]
    auto est_mags = masks * mix_mag;  // [B, C, F, T]

    // Perm 1
    auto l1_p1 = (est_mags.select(1,0) - tgt_mags.select(1,0)).abs().mean({-1,-2})
               + (est_mags.select(1,1) - tgt_mags.select(1,1)).abs().mean({-1,-2});
    // Perm 2
    auto l1_p2 = (est_mags.select(1,0) - tgt_mags.select(1,1)).abs().mean({-1,-2})
               + (est_mags.select(1,1) - tgt_mags.select(1,0)).abs().mean({-1,-2});

    return torch::min(l1_p1, l1_p2).mean();
}

}  // namespace losses
