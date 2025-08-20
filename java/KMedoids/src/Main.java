import java.io.IOException;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.logging.Level;

import elki.clustering.kmedoids.FasterPAM;
import elki.clustering.kmedoids.initialization.GreedyG;
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
	// Load N×N float matrix (little-endian) as a FloatBuffer
	static FloatBuffer mapFloatMatrix(Path path, int n) throws IOException {
		try (var channel = FileChannel.open(path)) {
			long need = (long)n * n * Float.BYTES;
			long have = channel.size();
			if (have != need) throw new IllegalArgumentException("File size mismatch: need " + need + " bytes, have " + have);
			var mb = channel.map(FileChannel.MapMode.READ_ONLY, 0, have);
			return mb.order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
		}
	}
	
	static Database buildDB(int n) {
		// Trivial 1D data; values are ignored because we use a DBID-based distance.
		double[][] data = new double[n][1];
		for (int i = 0; i < n; i++) {
			data[i][0] = 0.0;
		}
		var conn = new ArrayAdapterDatabaseConnection(data);
		Database db = new StaticArrayDatabase(conn, null);
		db.initialize();
		return db;
	}
	
	public static void main(String[] args) throws Exception {
		if (args.length < 3) {
			System.err.println("Usage: java -jar KMedoids.jar <path/to/emd.bin> <N> <k>");
			System.exit(2);
		}
		final Path binPath = Path.of(args[0]);
		final int N = Integer.parseInt(args[1]);
		final int K = Integer.parseInt(args[2]);
		
		LoggingConfiguration.setVerbose(Level.ALL);
		LoggingConfiguration.setDefaultLevel(Level.ALL);
		System.out.println("1) Load precomputed distances");
		FloatBuffer fb = mapFloatMatrix(binPath, N);
		
		System.out.println("2) Build DB with N objects");
		Database db = buildDB(N);
		
		System.out.println("3) Use the DBID relation (0..N-1)");
		Relation<DBID> rel = db.getRelation(TypeUtil.DBID);
		DBIDRange ids = elki.database.ids.DBIDUtil.assertRange(rel.getDBIDs());
		
		System.out.println("4) Distance function backed by precomputed matrix");
		var dist = new BufferDistance(fb, N);
		
		System.out.println("5) Run FasterPAM (exact k-medoids, fast build+swap)");
		var algo = new FasterPAM<>(dist, K, 5, new LAB<>(RandomFactory.DEFAULT));
		Clustering<MedoidModel> result = algo.run(rel);
		
		System.out.println("6) Extract medoids (as 0-based indices) and labels (cluster IDs)");
		List<Integer> medoids = new ArrayList<>(K);
		for (var c : result.getAllClusters()) {
			DBID med = c.getModel().getMedoid();
			medoids.add(ids.getOffset(med));
		}
		int[] labels = new int[N];
		int cid = 0;
		for (var c : result.getAllClusters()) {
			for (DBIDIter it = c.getIDs().iter(); it.valid(); it.advance()) {
				labels[ids.getOffset(it)] = cid;
			}
			cid++;
		}
		
		System.out.println("7) Print (or write to file)");
		System.out.println("Medoids (0-based):");
		for (int m : medoids) System.out.println(m);
		System.out.println("Labels (first 500):");
		for (int i = 0; i < Math.min(500, N); i++) System.out.println(labels[i]);
		
	}
}
