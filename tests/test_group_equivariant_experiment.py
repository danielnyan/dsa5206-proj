"""Controller and recovery tests with synthetic checkpoints/mocked subprocesses.

No real training or VAL2 image reads. The separate evaluator tests cover model
loading, inference, and metric export on synthetic images.
"""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import escnn
import torch
from PIL import Image

from modern_pca import run_group_equivariant_experiment as experiment


def metric(epoch, train_loss=.04, val_loss=.05, train_accuracy=.98, val_accuracy=.98):
    return dict(epoch=epoch, train_loss=train_loss, validation_loss=val_loss,
                train_accuracy=train_accuracy, validation_accuracy=val_accuracy)


class StoppingTests(unittest.TestCase):
    def test_requires_minimum_epochs_and_both_curves_stable(self):
        policy = experiment.DEFAULT_STOPPING
        self.assertFalse(experiment.stopping_decision([metric(i) for i in range(1, 10)], policy)['stop'])
        history = [metric(i) for i in range(1, 11)]
        self.assertEqual(experiment.stopping_decision(history, policy)['reason'], 'train_and_validation_stable')
        for name in experiment.METRICS:
            changed = copy.deepcopy(history)
            changed[-1][name] += .01
            self.assertFalse(experiment.stopping_decision(changed, policy)['stop'], name)

    def test_entire_window_range_rejects_slow_cumulative_drift(self):
        history = [metric(i) for i in range(1, 11)]
        for index, row in enumerate(history[-4:]):
            row['train_loss'] += index * .0008
        self.assertFalse(experiment.stopping_decision(history, experiment.DEFAULT_STOPPING)['stop'])

    def test_maximum_epoch_and_invalid_values(self):
        history = [metric(i, train_loss=1 / i) for i in range(1, 31)]
        self.assertEqual(experiment.stopping_decision(history, experiment.DEFAULT_STOPPING)['reason'], 'max_epochs')
        history[0]['train_loss'] = float('nan')
        with self.assertRaises(ValueError):
            experiment.stopping_decision(history, experiment.DEFAULT_STOPPING)
        with self.assertRaises(ValueError):
            experiment.stopping_decision([metric(2)], experiment.DEFAULT_STOPPING)
        with self.assertRaises(ValueError):
            experiment.validate_policy(dict(experiment.DEFAULT_STOPPING, loss_tolerance=-.1))


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.calls = []
        self.args = experiment.parse_args([
            '--data', str(self.root / 'data'), '--output', str(self.root / 'experiment'),
            '--device', 'cpu', '--min-epochs', '3', '--window', '3', '--max-epochs', '6'])
        self.contract = dict(
            batch_size=4, seed=42, torch=str(torch.__version__), escnn=escnn.__version__,
            preprocessing='RGB-bilinear350-minus1-plus1',
            model_sha256=experiment.sha256(experiment.ROOT / 'modern_pca/group_equivariant_model.py'),
            split={name: experiment.sha256(experiment.ROOT / 'manifests/VAL1_internal_v1' / name)
                   for name in ('VAL1_train.csv', 'VAL1_internal_validation.csv')})

    def save_checkpoint(self, path, epoch, history=None, next_batch=0):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(contract=self.contract, model={},
                        progress=dict(epoch=epoch, next_batch=next_batch,
                                      history=history or [metric(i) for i in range(1, epoch + 1)])), path)

    def fake_execute(self, command, log_path):
        def option(name):
            return command[command.index(name) + 1]
        if command[3] == 'modern_pca.train_group_equivariant':
            epoch = int(option('--epochs'))
            self.calls.append(('train', epoch))
            if '--resume' in command:
                previous = torch.load(option('--resume'), map_location='cpu', weights_only=False)
                self.assertEqual(previous['progress']['epoch'], epoch - 1)
            self.save_checkpoint(Path(option('--output')) / 'last.pt', epoch)
        else:
            self.assertEqual(command[3], 'modern_pca.evaluate_group_equivariant')
            source = experiment.checkpoint_info(option('--checkpoint'), 4)
            self.calls.append(('evaluate', source['epoch']))
            destination = Path(option('--output'))
            destination.mkdir(parents=True)
            experiment.write_json(destination / 'metrics.json', dict(
                status='complete', checkpoint_sha256=source['sha256'],
                completed_training_epochs=source['epoch'], samples=34103, cohort='VAL2',
                evaluator_sha256=experiment.sha256(experiment.ROOT / 'modern_pca/evaluate_group_equivariant.py'),
                manifest_sha256=experiment.sha256(experiment.ROOT / 'manifests/VAL2_manifest.csv'),
                metrics={'accuracy': .5}))
            (destination / 'predictions.csv').write_text('synthetic test export\n')

    def run_experiment(self, execute=None):
        with patch.object(experiment, 'execute', side_effect=execute or self.fake_execute):
            return experiment.run(self.args)

    def test_stability_selection_and_idempotent_completed_resume(self):
        result = self.run_experiment()
        self.assertEqual(self.calls, [('train', 1), ('train', 2), ('train', 3), ('evaluate', 3)])
        self.assertEqual(result['selection']['reason'], 'train_and_validation_stable')
        self.assertTrue((self.args.output / 'epochs/epoch_003.pt').exists())
        self.args.resume = True
        self.assertEqual(self.run_experiment(), result)
        self.assertEqual(len(self.calls), 4)

    def test_max_epoch_selects_last_not_best(self):
        def varying(command, log_path):
            if command[3] == 'modern_pca.train_group_equivariant':
                epoch = int(command[command.index('--epochs') + 1])
                self.calls.append(('train', epoch))
                destination = Path(command[command.index('--output') + 1]) / 'last.pt'
                self.save_checkpoint(destination, epoch, [metric(i, val_loss=.05 + i * .02) for i in range(1, epoch + 1)])
            else:
                self.fake_execute(command, log_path)
        result = self.run_experiment(varying)
        self.assertEqual(result['selection']['epoch'], 6)
        self.assertEqual(result['selection']['reason'], 'max_epochs')
        self.assertEqual(self.calls[-1], ('evaluate', 6))

    def test_partial_training_resume_and_protocol_lock(self):
        def interrupt(command, log_path):
            self.assertEqual(command[3], 'modern_pca.train_group_equivariant')
            destination = Path(command[command.index('--output') + 1]) / 'last.pt'
            self.save_checkpoint(destination, 0, next_batch=12)
            raise RuntimeError('simulated training interruption')
        with self.assertRaisesRegex(RuntimeError, 'interruption'):
            self.run_experiment(interrupt)
        self.assertFalse((self.args.output / 'selection.json').exists())
        self.assertFalse((self.args.output / 'val2').exists())
        self.args.resume = True
        self.args.loss_tolerance = .002
        with self.assertRaisesRegex(ValueError, 'Frozen protocol'):
            self.run_experiment()
        self.args.loss_tolerance = .001
        result = self.run_experiment()
        self.assertEqual(result['selection']['epoch'], 3)

    def test_completed_epoch_crash_recovers_archive_without_retraining(self):
        def interrupt(command, log_path):
            self.fake_execute(command, log_path)
            raise RuntimeError('simulated controller interruption')
        with self.assertRaises(RuntimeError):
            self.run_experiment(interrupt)
        self.args.resume = True
        self.run_experiment()
        self.assertEqual(self.calls.count(('train', 1)), 1)
        self.assertTrue((self.args.output / 'epochs/epoch_001.pt').exists())

    def test_partial_evaluation_retry_does_not_train_again(self):
        def interrupt(command, log_path):
            if command[3] == 'modern_pca.evaluate_group_equivariant':
                destination = Path(command[command.index('--output') + 1])
                destination.mkdir(parents=True)
                (destination / 'predictions.csv.partial').write_text('incomplete')
                raise RuntimeError('simulated evaluation interruption')
            self.fake_execute(command, log_path)
        with self.assertRaises(RuntimeError):
            self.run_experiment(interrupt)
        self.args.resume = True
        result = self.run_experiment()
        self.assertEqual(result['evaluation_report'], 'val2/attempt_002/metrics.json')
        self.assertEqual([call for call in self.calls if call[0] == 'train'], [('train', 1), ('train', 2), ('train', 3)])

    def test_completed_evaluation_reused_after_summary_interruption(self):
        def interrupt(command, log_path):
            self.fake_execute(command, log_path)
            if command[3] == 'modern_pca.evaluate_group_equivariant':
                raise RuntimeError('simulated interruption after evaluator completion')
        with self.assertRaises(RuntimeError):
            self.run_experiment(interrupt)
        self.args.resume = True
        self.run_experiment()
        self.assertEqual(self.calls.count(('evaluate', 3)), 1)

    def test_pilot_copy_is_immutable_and_plan_writes_nothing(self):
        pilot = self.root / 'pilot.pt'
        self.save_checkpoint(pilot, 1)
        original = pilot.read_bytes()
        self.args.resume_from = pilot
        self.args.plan = True
        self.run_experiment()
        self.assertFalse(self.args.output.exists())
        self.assertFalse(self.calls)
        self.args.plan = False
        self.run_experiment()
        self.assertEqual(self.calls[0], ('train', 2))
        self.assertEqual(pilot.read_bytes(), original)
        self.assertEqual((self.args.output / 'initial.pt').read_bytes(), original)
        self.args.resume = True
        self.args.resume_from = None
        with (self.args.output / 'final.pt').open('ab') as stream:
            stream.write(b'modification')
        with self.assertRaisesRegex(ValueError, 'Frozen checkpoint'):
            self.run_experiment()


class TrainerRecoveryTests(unittest.TestCase):
    def test_epochwise_resume_matches_uninterrupted_training_on_synthetic_images(self):
        from modern_pca import train_group_equivariant as trainer
        old_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        self.addCleanup(torch.set_num_threads, old_threads)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = []
            for index in range(4):
                label = 'benign' if index % 2 == 0 else 'tumour'
                directory = 'norm' if label == 'benign' else 'tu'
                relative = f'{directory}/{index}.jpg'
                image_path = root / 'data' / f'val_dataset_1_{directory}' / relative
                image_path.parent.mkdir(parents=True, exist_ok=True)
                Image.new('RGB', (32, 32), (30 + index * 40, 100, 150)).save(image_path)
                rows.append(dict(cohort='VAL1', class_name=label, ground_truth_code=index % 2,
                                 relative_path=f'{label}/{relative}', sample_id=f'VAL1:{label}/{relative}'))
            split = {'fingerprints': {'synthetic': 'test-only'}}

            def train(output, epochs, resume=None):
                args = SimpleNamespace(data=root / 'data', output=output, resume=resume,
                                       epochs=epochs, batch_size=2, checkpoint_every=100, device='cpu')
                with patch.object(trainer, 'parse_args', return_value=args), \
                     patch('modern_pca.reimplementation.freeze_split', return_value=(rows[:2], rows[2:], split)):
                    trainer.main()

            train(root / 'continuous', 2)
            train(root / 'resumed', 1)
            train(root / 'resumed', 2, root / 'resumed/last.pt')
            a = torch.load(root / 'continuous/last.pt', map_location='cpu', weights_only=False)
            b = torch.load(root / 'resumed/last.pt', map_location='cpu', weights_only=False)
            self.assertEqual(a['progress'], b['progress'])
            self.assertEqual(a['scheduler'], b['scheduler'])
            self.assertEqual(a['optimizer']['param_groups'], b['optimizer']['param_groups'])
            for key, value in a['model'].items():
                self.assertTrue(torch.equal(value, b['model'][key]), key)
            for key, state in a['optimizer']['state'].items():
                for name, value in state.items():
                    other = b['optimizer']['state'][key][name]
                    if isinstance(value, torch.Tensor):
                        self.assertTrue(torch.equal(value, other))
                    else:
                        self.assertEqual(value, other)


if __name__ == '__main__':
    unittest.main()
