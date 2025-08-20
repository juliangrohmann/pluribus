import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.logging.Level;

import elki.clustering.kmedoids.FasterPAM;
import elki.clustering.kmedoids.initialization.LAB;
import elki.data.Clustering;
import elki.data.model.MedoidModel;
import elki.data.type.TypeUtil;
import elki.database.Database;
import elki.database.StaticArrayDatabase;
import elki.database.ids.*;
import elki.database.relation.Relation;
import elki.datasource.ArrayAdapterDatabaseConnection;
import elki.logging.LoggingConfiguration;
import elki.utilities.random.RandomFactory;

public class Main {
	static int readIndexCount(Path path) throws IOException {
		try (var channel = FileChannel.open(path)) {
			long have = channel.size();
			return (int)Math.sqrt((double)have / Float.BYTES);
		}
	}
	
	static Database buildDB(int n) {
		double[][] data = new double[n][1];
		for (int i = 0; i < n; i++) {
			data[i][0] = 0.0;
		}
		var conn = new ArrayAdapterDatabaseConnection(data);
		Database db = new StaticArrayDatabase(conn, null);
		db.initialize();
		return db;
	}
	
	public static void writeIntArray(Path path, int[] data) throws IOException {
		ByteBuffer buf = ByteBuffer.allocate(data.length * Integer.BYTES);
		buf.order(ByteOrder.LITTLE_ENDIAN);   // match C++ side
		for (int v : data) {
			buf.putInt(v);
		}
		buf.flip();
		Files.write(path, buf.array());
	}
	
	public static void main(String[] args) throws Exception {
		if (args.length < 3) {
			System.err.println("Usage: java -jar KMedoids.jar <start_flop> <end_flop> <k> <path/to/labels.bin>");
			System.exit(2);
		}
		final int start = Integer.parseInt(args[0]);
		final int end = Integer.parseInt(args[1]);
		final int K = Integer.parseInt(args[2]);
		final Path baseDir = Path.of(args[3]);
		
		LoggingConfiguration.setVerbose(Level.WARNING);
		LoggingConfiguration.setDefaultLevel(Level.WARNING);
		
		for(int flop_idx = start; flop_idx < end; ++flop_idx) {
			System.out.println("Clustering flop " + flop_idx);
			final Path emdPath = baseDir.resolve("emd_matrix_r2_f" + flop_idx + "_c" + K + ".bin");
			final int N = readIndexCount(emdPath);
			System.out.println("Indexes: " + N);
			RowStripeMapper map = new RowStripeMapper(emdPath, N);
			Database db = buildDB(N);
			Relation<DBID> rel = db.getRelation(TypeUtil.DBID);
			DBIDRange ids = elki.database.ids.DBIDUtil.assertRange(rel.getDBIDs());
			var dist = new BufferDistance(map);
			var algo = new FasterPAM<>(dist, K, 1000, new LAB<>(RandomFactory.DEFAULT));
			System.out.println("Clustering...");
			long t0 = System.nanoTime();
			Clustering<MedoidModel> result = algo.run(rel);
			double dt = t0 - System.nanoTime() / 1000.0 / 1000.0 / 1000.0;
			System.out.println("Clustered in " + dt + " s");
			
			int[] labels = new int[N];
			int cid = 0;
			for (var c : result.getAllClusters()) {
				for (DBIDIter it = c.getIDs().iter(); it.valid(); it.advance()) {
					labels[ids.getOffset(it)] = cid;
				}
				cid++;
			}
			var out_fn = baseDir.resolve("clusters_r2_f" + flop_idx + "_c" + K + ".bin");
			System.out.println("Saving clusters to " + out_fn);
			writeIntArray(out_fn, labels);
		}
	}
}
