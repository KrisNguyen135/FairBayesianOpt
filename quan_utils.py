from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from aif360.datasets import CompasDataset


def generate_fairness_report(dataset, privileged_groups, unprivileged_groups):
    print(f'Shape: {dataset.features.shape}\n')

    print(f'Favorable label: {dataset.favorable_label}')
    print(f'Unfavorable label: {dataset.unfavorable_label}\n')

    print('Protected attribute names:')
    print(dataset.protected_attribute_names)
    print()

    print('Privileged attribute values:')
    print(dataset.privileged_protected_attributes)
    print('Unprivileged attribute values:')
    print(dataset.unprivileged_protected_attributes)
    print()

    binary_label_metric = BinaryLabelDatasetMetric(
        dataset,
        privileged_groups=privileged_groups,
        unprivileged_groups=unprivileged_groups
    )
    print(f'Statistical parity difference: '
          f'{binary_label_metric.statistical_parity_difference()}')
    print(f'Disparate impact: {binary_label_metric.disparate_impact()}')

    # classification_metric = ClassificationMetric(
    #     dataset,
    #     privileged_groups=privileged_groups,
    #     unprivileged_groups=unprivileged_groups
    # )
    # print(f'Equal opportunity difference: '
    #       f'{classification_metric.equal_opportunity_difference()}')


if __name__ == '__main__':
    generate_fairness_report(
        CompasDataset(),
        privileged_groups=[{'race': 1}],
        unprivileged_groups=[{'race': 0}]
    )
