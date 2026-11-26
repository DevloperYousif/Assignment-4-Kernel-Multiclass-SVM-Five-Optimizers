# Academic Defense Guide - Assignment 4 (Kernel Multi-Class SVM)

دليل الدفاع الأكاديمي: أسئلة متوقعة من الدكتور مع أجوبة بالعربي (مع المصطلحات بالإنجليزي) وجواب قصير بالإنجليزي.

**Key numbers to remember / أرقام لازم تحفظها:**

- Dataset: UCI HAR, 6 classes, 561 features, train 5,551 (16 subjects) / validation 1,801 (5) / test 2,947 (9)
- RBF γ = 4 × γ₀ = 4.310e-03, λ = 0.0001, basis M = 2,000, 100 epochs
- Test accuracy: Gradient Descent 86.46%, SGD 93.59%, Mini-Batch GD 92.37%, AdaGrad 93.82%, Adam 92.70%
- Linear SVM 94.91% · Kernel SVM (Kernel SVM (RBF, Adam)) 92.70% · sklearn SVC 94.60%

## Q1. What is SVM?
**السؤال:** شو هو الـ SVM؟

**الجواب:** الـ Support Vector Machine هو classifier بيدور على الحد الفاصل (hyperplane) اللي إله أكبر margin بين الـ classes. النقاط القريبة من الحد اسمها support vectors وهي اللي بتحدد الحل. بالـ soft margin بنسمح بأغلاط ونعاقبها بالـ hinge loss.

**Short answer (EN):** A maximum-margin classifier; the closest points (support vectors) define the boundary; soft margin uses the hinge loss.

## Q2. What is a kernel?
**السؤال:** شو هو الـ kernel؟

**الجواب:** دالة k(x, z) بتعطي الـ inner product بين صورتين للعينات بـ feature space: k(x,z) = ⟨φ(x), φ(z)⟩، بدون ما نحسب φ. مثال: RBF، polynomial، linear.

**Short answer (EN):** A similarity function equal to an inner product in some feature space, k(x,z) = <phi(x), phi(z)>.

## Q3. What is the kernel trick?
**السؤال:** شو هو الـ kernel trick؟

**الجواب:** بما إنه الـ SVM بيحتاج بس inner products، بنبدلها بالـ kernel. هيك بنشتغل كأنه بـ feature space عالي الأبعاد (أو لا نهائي بالـ RBF) بدون ما نبنيه. مثال: (xᵀz)² ببعدين = inner product بـ 3 أبعاد.

**Short answer (EN):** Replace inner products by k(x,z) to work implicitly in a high/infinite-dimensional space without building it.

## Q4. Why do we need nonlinear classification?
**السؤال:** ليش بنحتاج تصنيف non-linear؟

**الجواب:** لأنه بعض الداتا ما بتنفصل بخط مستقيم، زي الحلقات بالمثال الـ synthetic (Linear SVM فشل والـ RBF نجح). بالـ HAR ممكن الحدود بين القعدة والوقفة تكون منحنية.

**Short answer (EN):** Some classes (e.g. concentric rings) are not linearly separable in the input space.

## Q5. What is the difference between Linear SVM and Kernel SVM?
**السؤال:** شو الفرق بين Linear SVM و Kernel SVM؟

**الجواب:** الـ Linear حده hyperplane بالـ input space وعدد معاملاته d × K (561 × 6). الـ Kernel حده منحني، والموديل مجموع kernels على basis (عنا 2,000 × 6 معامل + تخزين الـ basis). بنتائجنا: Linear 94.91% و Kernel 92.70%.

**Short answer (EN):** Linear: hyperplane, O(dK) parameters. Kernel: curved boundary as a kernel expansion over a basis, more expensive.

## Q6. What is the RBF kernel?
**السؤال:** شو هو الـ RBF kernel؟

**الجواب:** k(x,z) = exp(−γ‖x−z‖²). قيمته 1 لنفس النقطة وبتنزل مع المسافة، يعني تأثير محلي. دايماً PSD وقيمه بين 0 و 1.

**Short answer (EN):** Gaussian similarity exp(-gamma ||x-z||^2): local, bounded, always PSD.

## Q7. What is the Polynomial kernel?
**السؤال:** شو هو الـ Polynomial kernel؟

**الجواب:** k(x,z) = (γxᵀz + r)ᵈ: بيمثل تفاعلات بين الـ features لحد الدرجة d. جربنا d = 2 (test Macro F1 0.9001) و d = 3 (0.8889).

**Short answer (EN):** (gamma x^T z + r)^d - all feature interactions up to degree d.

## Q8. What is gamma?
**السؤال:** شو هو الـ gamma؟

**الجواب:** بالـ RBF هو مقلوب مربع طول التأثير: بيحدد كم بسرعة بيقل التشابه مع المسافة. بلشنا بالـ median heuristic γ₀ = 1.077e-03 واخترنا على الـ validation γ = 4γ₀ = 4.310e-03.

**Short answer (EN):** Inverse squared length-scale of the RBF; we used gamma = 4 x median heuristic.

## Q9. What happens when gamma is too small?
**السؤال:** شو بيصير لو الـ gamma صغيرة كثير؟

**الجواب:** كل نقطة بتأثر على مسافات بعيدة، والـ kernel بيصير شبه ثابت، فالحدود ناعمة كثير وشبه خطية: underfitting. عنا γ = 0.5γ₀ أعطت validation Macro F1 أقل بوضوح.

**Short answer (EN):** Very smooth, almost linear functions - underfitting.

## Q10. What happens when gamma is too large?
**السؤال:** شو بيصير لو الـ gamma كبيرة كثير؟

**الجواب:** كل نقطة بتأثر بس على جيرانها القريبين جداً، والـ G بتقرب من مصفوفة الوحدة، فالموديل بيحفظ التدريب وما بيعمم: overfitting. عنا 8γ₀ كانت أسوأ من 4γ₀، و 64γ₀ انهارت تماماً بالتجارب الأولية.

**Short answer (EN):** Very local bumps, G -> I, memorisation - overfitting.

## Q11. What is a Gram matrix?
**السؤال:** شو هي الـ Gram matrix؟

**الجواب:** مصفوفة Gᵢⱼ = k(zᵢ, zⱼ) فيها كل التشابهات بين نقاط الـ basis. عنا حجمها 2,000 × 2,000، symmetric، قطرها 1 بالـ RBF.

**Short answer (EN):** The matrix of pairwise kernel values G_ij = k(z_i, z_j).

## Q12. Why must a valid kernel generally be positive semidefinite?
**السؤال:** ليش الـ kernel لازم يكون PSD؟

**الجواب:** لأنه PSD معناه إنه فعلاً inner product بـ feature space (شرط Mercer)، فالـ norm aᵀGa ≥ 0، والـ objective بيضل convex وإله حل أمثل وحيد. الـ sigmoid عنا أعطى eigenvalue سالبة (-49.2)، فاستخدمناه للاستكشاف بس.

**Short answer (EN):** PSD = genuine inner product, non-negative RKHS norm, convex objective.

## Q13. What is RKHS?
**السؤال:** شو هو الـ RKHS؟

**الجواب:** Reproducing Kernel Hilbert Space: فضاء الدوال f = Σ aⱼ k(·, zⱼ)، والـ norm تبعها ‖f‖² = aᵀGa، وفيه خاصية reproducing: f(x) = ⟨f, k(·,x)⟩. الـ norm الصغيرة يعني دالة ناعمة.

**Short answer (EN):** The Hilbert space of functions generated by the kernel, with ||f||^2 = a^T G a.

## Q14. What is the Representer Theorem?
**السؤال:** شو هي الـ Representer Theorem؟

**الجواب:** بتحكي إنه حل مسألة (loss على نقاط التدريب + λ‖f‖²) بيكون مجموع kernels متمركزة على نقاط التدريب. عشان هيك بنتعلم مصفوفة A بدل w لا نهائي.

**Short answer (EN):** The optimal f is a finite kernel expansion over training points, so we optimise coefficients A.

## Q15. How does your model handle multiple classes?
**السؤال:** كيف الموديل بيتعامل مع أكثر من class؟

**الجواب:** لكل class دالة fₖ، والـ scores S = HA + b بحجم N × 6، والتوقع argmax. كل الـ classes بمسألة وحدة مشتركة (direct multi-class).

**Short answer (EN):** One score function per class trained jointly; predict the argmax.

## Q16. Why did you use Crammer-Singer hinge loss?
**السؤال:** ليش استخدمت Crammer-Singer hinge loss؟

**الجواب:** لأنه طريقة مباشرة: موديل واحد بدل 6 أو 15، وبيطلب من الـ class الصح يغلب أقوى class غلط بفرق 1. وهو المطلوب بالأسايمنت، وبيعطي objective واحد نقدر نقارن عليه الخمس optimizers.

**Short answer (EN):** A single joint multi-class margin loss - one objective for all classes and all optimizers.

## Q17. How is your kernelized objective calculated?
**السؤال:** كيف بينحسب الـ objective؟

**الجواب:** J = (λ/2)Tr(AᵀGA) + (1/N)Σ max(0, 1 + max_(j≠yᵢ) Sᵢⱼ − Sᵢ,yᵢ). مثلاً للـ SGD: hinge 0.0993 + reg 0.0397 = 0.1391.

**Short answer (EN):** J = (lambda/2) Tr(A^T G A) + mean Crammer-Singer hinge loss.

## Q18. How is RKHS regularization calculated?
**السؤال:** كيف بتنحسب الـ RKHS regularization؟

**الجواب:** (λ/2)·Tr(AᵀGA) = (λ/2)·Σₖ aₖᵀGaₖ = مجموع ‖fₖ‖² بالـ RKHS. بالكود: 0.5·λ·sum(A * (G @ A)). مش ‖A‖² العادية.

**Short answer (EN):** (lambda/2) sum_k a_k^T G a_k - the kernel norm, not ||A||^2.

## Q19. How are the model subgradients calculated?
**السؤال:** كيف بتنحسب الـ subgradients؟

**الجواب:** بنبني D: للعينة اللي فيها violation بنحط −1 عند الـ class الصح و +1 عند أقوى class غلط، والباقي صفر. بعدين ∂J/∂A = HᵀD/N + λGA و ∂J/∂b = ΣD/N. تأكدنا منها بـ finite differences بالـ tests.

**Short answer (EN):** D = -1 at true class, +1 at strongest wrong class for violators; dJ/dA = H^T D/N + lambda G A.

## Q20. Why is feature scaling necessary?
**السؤال:** ليش الـ scaling ضروري؟

**الجواب:** الـ RBF بيعتمد على المسافة الإقليدية، فلو feature مداها أكبر بتسيطر على المسافة. الـ StandardScaler بيخلي كل feature متوسطها 0 وانحرافها 1، ومحسوب من الـ train بس.

**Short answer (EN):** RBF uses Euclidean distance; unscaled large-range features would dominate.

## Q21. How did you implement GD?
**السؤال:** كيف طبقت الـ GD؟

**الجواب:** بكل epoch بنحسب الـ gradient على كل الـ 5,551 عينة ونعمل update وحدة: θ ← θ − ηg. الـ lr المختار 1، و 100 update بس، فالنتيجة 86.46%.

**Short answer (EN):** One full-batch update per epoch.

## Q22. How did you implement SGD?
**السؤال:** كيف طبقت الـ SGD؟

**الجواب:** عينة وحدة بكل update بترتيب عشوائي ثابت (seed 42)، يعني 555,100 update. وعشان السرعة استخدمنا C = UᵀA من الـ eigendecomposition تبع G، وهاد بيعطي نفس الـ iterates بالزبط بس بتكلفة O(MK).

**Short answer (EN):** Batch size 1, reproducible shuffling, exact rotated update (verified equal to naive SGD).

## Q23. How did you implement Mini-Batch GD?
**السؤال:** كيف طبقت الـ Mini-Batch GD؟

**الجواب:** batch 64، 87 update بالـ epoch، وآخر batch فيه 47 عينة. الـ gradient متوسط الـ batch + λGA كامل.

**Short answer (EN):** Average subgradient over 64 samples plus the exact regularisation gradient; last batch smaller.

## Q24. How does AdaGrad work?
**السؤال:** كيف بيشتغل الـ AdaGrad؟

**الجواب:** بيجمع مربعات الـ gradients لكل معامل (sq_grad_sum)، وبيقسم الخطوة على الجذر: θ ← θ − ηg/(√s + ε). المعاملات اللي gradients تبعها كبيرة بتاخد خطوات أصغر، والـ effective lr بيقل مع الوقت.

**Short answer (EN):** Per-parameter step size eta / sqrt(sum of squared gradients).

## Q25. How does Adam work?
**السؤال:** كيف بيشتغل الـ Adam؟

**الجواب:** m = متوسط متحرك للـ gradients (momentum)، v = متوسط متحرك لمربعاتها، مع bias correction لأنهم بيبلشوا من صفر، والخطوة η·m̂/(√v̂ + ε). β₁ = 0.9، β₂ = 0.999.

**Short answer (EN):** Momentum + RMSProp scaling + bias correction.

## Q26. Why did you implement optimizers manually?
**السؤال:** ليش كتبت الـ optimizers يدوي؟

**الجواب:** لأنه المطلوب بالأسايمنت، وعشان نفهم كل خطوة، وعشان كل الخمسة يشتغلوا على نفس الـ objective بالزبط بدون اختلافات مكتبات. وكل update rule عليه unit test.

**Short answer (EN):** Required, transparent, and guarantees the identical objective; each rule is unit-tested.

## Q27. Why does Scikit-learn SVC not expose an Adam optimizer parameter?
**السؤال:** ليش SVC ما فيه parameter للـ Adam؟

**الجواب:** لأنه SVC بيحل الـ dual problem (quadratic programming مع قيود) بخوارزمية SMO من libsvm، مش بـ gradient descent. فما في learning rate ولا optimizer تختاره.

**Short answer (EN):** SVC solves the constrained dual QP with SMO (libsvm), not with gradient-based optimizers.

## Q28. What is the difference between SVC and your implementation?
**السؤال:** شو الفرق بين SVC والتطبيق تبعك؟

**الجواب:** SVC: dual QP + SMO + One-vs-One (15 موديل) + كل نقاط التدريب ممكن تكون support vectors. إحنا: primal بدون قيود بالـ coefficient space + Crammer-Singer مباشر + basis من 2,000 + first-order optimizers. النتيجة: SVC 94.60% وإحنا 92.70%.

**Short answer (EN):** Dual QP/SMO/OvO versus unconstrained primal coefficient-space direct multi-class with first-order methods.

## Q29. What is the difference between One-vs-One and One-vs-Rest?
**السؤال:** شو الفرق بين OvO و OvR؟

**الجواب:** OvR: K موديلات، كل class ضد الباقي. OvO: K(K−1)/2 موديل، كل زوج لحاله، والتوقع بالتصويت. لـ 6 classes: 6 مقابل 15.

**Short answer (EN):** OvR trains K one-vs-all models; OvO trains K(K-1)/2 pairwise models with voting.

## Q30. What is direct multi-class SVM?
**السؤال:** شو هو الـ direct multi-class SVM؟

**الجواب:** مسألة optimization وحدة فيها كل الـ classes مع بعض (زي Crammer-Singer)، بدل ما نقسمها لمسائل binary.

**Short answer (EN):** A single joint optimisation over all classes.

## Q31. How did you prevent data leakage?
**السؤال:** كيف منعت الـ data leakage؟

**الجواب:** التقسيم حسب الشخص بـ GroupShuffleSplit، والـ scaler والـ basis والـ gamma كلهم من الـ train بس، والـ test ما استخدمناه بأي اختيار، ورقم الشخص مش feature.

**Short answer (EN):** Subject-disjoint splits; scaler, basis and gamma fitted on training data; test used once.

## Q32. How did you ensure a fair comparison?
**السؤال:** كيف ضمنت إنه المقارنة عادلة؟

**الجواب:** نفس الداتا والتقسيم والـ kernel والـ gamma والـ basis والبداية (A = 0) والـ lambda والـ objective و100 epoch. والـ learning rate لكل واحد من 5 قيم بنفس الميزانية على الـ validation.

**Short answer (EN):** Everything identical except the optimizer; equal tuning budget per optimizer.

## Q33. Why did you choose Macro F1-score?
**السؤال:** ليش اخترت Macro F1؟

**الجواب:** لأنه بيعطي كل نشاط نفس الوزن، فما بيخبي ضعف الموديل على class معين، والداتا فيها عدم توازن خفيف.

**Short answer (EN):** Every class counts equally.

## Q34. What is the time complexity of kernel matrix construction?
**السؤال:** شو الـ time complexity لبناء الـ kernel matrix؟

**الجواب:** O(N·M·d): لكل زوج (عينة، basis) بنحسب مسافة بـ d بُعد. للـ exact (M = N) بتصير O(N²d).

**Short answer (EN):** O(N M d), i.e. O(N^2 d) for the exact kernel.

## Q35. What is the memory complexity of a full Gram matrix?
**السؤال:** شو الـ memory complexity للـ Gram matrix الكاملة؟

**الجواب:** O(N²): لـ N = 5,551 بـ float64 حوالي 0.23 GiB، ولـ مليون عينة بتصير 8 TB، وهون المشكلة.

**Short answer (EN):** O(N^2) - 0.23 GiB here, infeasible for very large N.

## Q36. Why can Kernel SVM be slower than Linear SVM?
**السؤال:** ليش الـ Kernel SVM أبطأ من الـ Linear؟

**الجواب:** لأنه لازم نحسب الـ kernel بين كل عينة وكل الـ basis، ونخزن الـ basis، وكل update فيه GA بتكلفة M²K. عنا وقت التوقع مع الـ kernel 0.211 s مقابل 0.0031 s للـ linear.

**Short answer (EN):** Kernel evaluations against the basis, basis storage and O(M^2 K) regularisation gradients.

## Q37. What is Nyström approximation?
**السؤال:** شو هو الـ Nyström approximation؟

**الجواب:** بنختار M landmarks، وبنحسب φ(x) = k(x,Z)·U·Λ^(−1/2) من الـ eigendecomposition تبع G، فبيصير φ(x)ᵀφ(z) ≈ k(x,z). بعدين بندرب linear SVM على φ. ولنفس الـ basis هو re-parametrization لموديلنا.

**Short answer (EN):** Low-rank kernel approximation from landmark points: phi(x) = k(x,Z) U Lambda^(-1/2).

## Q38. What are Random Fourier Features?
**السؤال:** شو هي الـ Random Fourier Features؟

**الجواب:** للـ RBF: ω من N(0, 2γI) و u من U[0, 2π]، و φ(x) = √(2/D)·cos(xω + u)، فبيصير متوسط φ(x)ᵀφ(z) = k(x,z). جربناها وطلع الخطأ النسبي أعلى من Nyström بنفس الحجم.

**Short answer (EN):** Random cosine features whose inner product approximates the RBF kernel in expectation.

## Q39. Is your implementation exact or approximate?
**السؤال:** هل التطبيق exact ولا approximate؟

**الجواب:** الموديل الأساسي Mode B: kernel expansion حقيقية بـ RBF بس على basis مختصر من 2,000 عينة، يعني restricted مش approximate ومش exact كامل. قسنا الـ exact (M = 5,551): validation Macro F1 0.9406 مقابل 0.9241 للـ 2,000.

**Short answer (EN):** Restricted exact-kernel expansion (M = 2,000); the full exact basis was measured separately.

## Q40. Which optimizer achieved the best results?
**السؤال:** أي optimizer أعطى أحسن نتيجة؟

**الجواب:** حسب قاعدة الاختيار (validation Macro F1): Adam (0.9450). على الـ test أعلى accuracy AdaGrad (93.82%) وأعلى Macro F1 AdaGrad (0.9365). الأربعة الأوائل متقاربين، والـ GD الأضعف (86.46%).

**Short answer (EN):** Validation: Adam; test accuracy: AdaGrad; the top four are close, GD is last.

## Q41. Did Kernel SVM outperform Linear SVM?
**السؤال:** هل الـ Kernel SVM أحسن من الـ Linear؟

**الجواب:** لا بهاي التجربة: Linear 94.91% / 0.9486 مقابل Kernel 92.70% / 0.9252. الـ features مصممة بقوة وشبه linearly separable، والـ test ناس جداد، والموديل تبعنا محدود بالـ basis والـ epochs.

**Short answer (EN):** No - on HAR the controlled Linear SVM was at least as accurate and much cheaper.

## Q42. What are the project's limitations?
**السؤال:** شو قيود المشروع؟

**الجواب:** basis مختصر، 100 epoch بتظلم الـ GD، 5 أشخاص بس بالـ validation، seed واحد، ما في learning rate decay، والـ kernel selection بـ solver واحد.

**Short answer (EN):** Restricted basis, fixed budget, small validation set, single seed, no schedules.

## Q43. How does SVM differ from Logistic Regression?
**السؤال:** شو الفرق بين SVM و Logistic Regression؟

**الجواب:** الاثنين linear classifiers بالأساس. الـ LR بيصغر log-loss وبيعطي احتمالات، وكل النقاط بتأثر. الـ SVM بيصغر hinge loss، بيهتم بالـ margin، والنقاط البعيدة ما بتأثر (sparse)، وما بيعطي احتمالات مباشرة.

**Short answer (EN):** Log-loss and probabilities versus hinge loss, margin and sparse support vectors.

## Q44. How does SVM differ from Random Forest?
**السؤال:** شو الفرق بين SVM و Random Forest؟

**الجواب:** الـ Random Forest مجموعة decision trees بتقسم الـ features بقواعد if-else وما بتحتاج scaling. الـ SVM موديل margin-based بيعتمد على المسافات/الـ inner products وبيحتاج scaling، ومسألته convex.

**Short answer (EN):** Tree ensemble with axis-aligned splits versus convex margin-based kernel model.

## Q45. How could this project be improved?
**السؤال:** كيف ممكن نحسن المشروع؟

**الجواب:** Nyström features لـ conditioning أحسن، learning rate decay و early stopping، cross-validation حسب الأشخاص مع أكثر من seed، calibration للاحتمالات، و float32/GPU للـ exact kernel.

**Short answer (EN):** Better conditioning (Nystroem), schedules, subject-wise CV, calibration, faster exact kernels.

## Q46. Why is plain GD so slow on this problem?
**السؤال:** ليش الـ GD بطيء كثير هون؟

**الجواب:** لأنه بـ 100 epoch بيعمل 100 update بس، ولأنه أعمدة الـ RBF kernel مترابطة، فالـ HᵀH/N إلها eigenvalue كبيرة وحدة والباقي صغير جداً: الخطوة لازم تكون صغيرة عشان ما ينفجر، فالتقدم بالاتجاهات الصغيرة بطيء. جربنا lr = 3 وانفجر.

**Short answer (EN):** Only 100 updates and an ill-conditioned coefficient-space problem.

## Q47. Why did the validation winner differ from the test winner?
**السؤال:** ليش الفائز على الـ validation غير الفائز على الـ test؟

**الجواب:** لأنه الـ validation فيه 5 أشخاص بس والـ test 9 أشخاص مختلفين، والفروقات بين الأوائل حوالي نقطة. هاد noise من اختلاف الأشخاص، مش فرق حقيقي كبير.

**Short answer (EN):** Small validation set (5 subjects); differences of about one point are within noise.

## Q48. Why is the bias not regularized?
**السؤال:** ليش ما عاقبت الـ bias؟

**الجواب:** لأنه الـ bias بس بيزيح scores الـ class كلها، وما إله علاقة بنعومة الدالة. لو عاقبناه بنجبر الـ classes تكون متساوية مسبقاً بدون سبب.

**Short answer (EN):** The bias only shifts a class score; it does not affect smoothness.

## Q49. Why did you use a validation set instead of the test set for tuning?
**السؤال:** ليش استخدمت validation للـ tuning؟

**الجواب:** لو اخترنا الـ learning rate أو الـ gamma على الـ test بتصير النتيجة متفائلة وغير صادقة. الـ test لازم يضل مستقل للتقييم النهائي بس.

**Short answer (EN):** Tuning on the test set would bias the final estimate.
